from __future__ import annotations

import gc
import glob
import json
import os
import pickle
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from operator import attrgetter
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from data_manager.dataset_management import generate_image_name
from imutils.paths import list_images
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection.losses import focal_tversky, iou_seg
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from path_desc import _DIR_APP_NAME, _OLD_DIR_APP_NAME, BASE_DATA_DIR
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from stqdm import stqdm
from streamlit import session_state

if TYPE_CHECKING:
    from training.training_management import AugmentationConfig


def check_unique_label_counts(labels: List[int], encoded_label_dict: Dict[int, str]):
    """Check whether there is any class that has only 1 image, which will not work
    when trying to split with train_test_split() with stratification."""
    unique_values, counts = np.unique(labels, return_counts=True)
    one_member_idxs = (counts == 1)
    if one_member_idxs.any():
        label_idx = int(unique_values[one_member_idxs][0])
        label_name = encoded_label_dict[label_idx]
        logger.warning(
            f"The least populated class in y: '{label_name}' has only 1 image, "
            "which is too few. The minimum number of groups for any class cannot be less "
            "than 2. This will not work for stratification in `train_test_split()`, "
            "therefore, changing to no stratification.")
        return False
    return True


def get_class_distribution(
        train_labels: List[str], val_labels: List[str] = None,
        test_labels: List[str] = None,
        encoded_label_dict: Dict[int, str] = None) -> pd.DataFrame:
    train_label_cnts = Counter(train_labels)
    sorted_train_label_cnts = dict(sorted(train_label_cnts.items()))
    cnt_dict = {"Training Set": sorted_train_label_cnts}

    if val_labels is not None:
        val_label_cnts = Counter(val_labels)
        cnt_dict["Validation Set"] = val_label_cnts
    if test_labels is not None:
        test_label_cnts = Counter(test_labels)
        cnt_dict["Test Set"] = test_label_cnts

    df = pd.DataFrame(cnt_dict)
    df.index.rename('Classes', inplace=True)
    if encoded_label_dict:
        # change integers to class labels
        df.index = df.index.map(lambda x: encoded_label_dict[x])
    return df


def custom_train_test_split(image_paths: List[Path],
                            test_size: float,
                            *,
                            no_validation: bool,
                            labels: Union[List[str], List[Path]],
                            val_size: Optional[float] = 0.0,
                            train_size: Optional[float] = 0.0,
                            stratify: Optional[bool] = False,
                            show_class_distribution: Optional[bool] = False,
                            encoded_label_dict: Optional[Dict[int, str]] = None,
                            random_seed: Optional[int] = None
                            ) -> Tuple[List[str], ...]:
    """
    Splitting the dataset into train set, test set, and optionally validation set
    if `no_validation` is True. Image classification will pass in label names instead
    of label_paths for each image.

    Args:
        image_paths (Path): Directory to the images.
        test_size (float): Size of test set in percentage
        no_validation (bool): If True, only split into train and test sets, without validation set.
        labels (Union[str, Path]): Pass in this parameter to split the labels or label paths.
        val_size (Optional[float]): Size of validation split, only needed if `no_validation` is False. Defaults to 0.0.
        train_size (Optional[float]): This is only used for logging, can be inferred, thus not required. Defaults to 0.0.
        stratify (Optional[bool]): stratification should only be used for image classification. Defaults to False
        show_class_distribution (Optional[bool]): whether to show class distribution in a Streamlit table. Defaults to False.
        random_seed (Optional[int]): random seed to use for splitting. Defaults to None.

    Returns:
        Tuples of lists of image paths (str), and optionally annotation paths,
        optionally split without validation set too.
    """
    if no_validation:
        assert not val_size, "Set `no_validation` to True if want to split into validation set too."
    else:
        assert val_size, "Must pass in `val_size` if `no_validation` is False."

    total_images = len(image_paths)
    assert total_images == len(labels)

    if stratify:
        logger.info("Using stratification for train_test_split()")
        depl_type = session_state.new_training.deployment_type
        assert depl_type == 'Image Classification', (
            'Only use stratification for image classification labels. '
            f'Current deployment type: {depl_type}'
        )
        stratify = labels
    else:
        stratify = None

    logger.info(f"Total images = {total_images}")

    if no_validation:
        train_size = train_size if train_size else round(1 - test_size, 2)
        logger.info("Splitting into train:test dataset"
                    f" with ratio of {train_size:.2f}:{test_size:.2f}")
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels,
            test_size=test_size,
            stratify=stratify,
            random_state=random_seed
        )

        if show_class_distribution:
            df = get_class_distribution(
                train_labels=y_train, test_labels=y_test,
                encoded_label_dict=encoded_label_dict)
            st.subheader("Class Distribution")
            st.table(df)

        return X_train, X_test, y_train, y_test
    else:
        train_size = train_size if train_size else round(
            1 - test_size - val_size, 2)
        logger.info("Splitting into train:valid:test dataset"
                    " with ratio of "
                    f"{train_size:.2f}:{val_size:.2f}:{test_size:.2f}")
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            image_paths, labels,
            test_size=(val_size + test_size),
            stratify=stratify,
            random_state=random_seed
        )
        logger.debug(f"{len(X_train) = }, {len(y_train) = }, "
                     f"{len(X_val_test) = }, {len(y_val_test) = }")

        stratify = y_val_test if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=(test_size / (val_size + test_size)),
            # shuffle must be True if stratify is True
            shuffle=True if stratify else False,
            stratify=stratify,
            random_state=random_seed,
        )

        if show_class_distribution:
            df = get_class_distribution(
                train_labels=y_train, val_labels=y_val, test_labels=y_test,
                encoded_label_dict=encoded_label_dict)
            st.subheader("Class Distribution")
            st.table(df)

        return X_train, X_val, X_test, y_train, y_val, y_test


def copy_images(image_paths: Path,
                dest_dir: Path,
                label_paths: Optional[Path] = None):
    if dest_dir.exists():
        # remove the existing images
        shutil.rmtree(dest_dir, ignore_errors=False)

    # create new directories
    os.makedirs(dest_dir)

    if label_paths:
        for image_path, label_path in stqdm(zip(image_paths, label_paths), total=len(image_paths)):
            # copy the image file and label file to the new directory
            shutil.copy2(image_path, dest_dir)
            shutil.copy2(label_path, dest_dir)
    else:
        for image_path in image_paths:
            shutil.copy2(image_path, dest_dir)


def load_image_into_numpy_array(path: str, bgr2rgb: bool = True):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    # always read in 3 channels
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert from OpenCV's BGR to RGB format
    if bgr2rgb:
        # NOTE: This step is required for TFOD!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_transform(augmentation_config: AugmentationConfig, deployment_type: str) -> A.Compose:
    """Get the Albumentations' transform using the existing augmentation config stored in DB."""
    existing_aug = augmentation_config.augmentations

    transform_list = []
    for transform_name, param_values in existing_aug.items():
        transform_list.append(getattr(A, transform_name)(**param_values))

    if deployment_type == 'Object Detection with Bounding Boxes':
        min_area = augmentation_config.min_area
        min_visibility = augmentation_config.min_visibility
        transform = A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['class_names']
            ))
    else:
        transform = A.Compose(transform_list)
    return transform


def preprocess_image(
        image: np.ndarray, image_size: int,
        bgr2rgb: bool = True, preprocess_fn: Callable = None,
        rescale: bool = True) -> np.ndarray:
    """Please note that this function should follow the preprocessing function 
    used during training."""
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # using INTER_NEAREST_EXACT to be identical with tensorflow's 'nearest' method
    image = cv2.resize(image, (image_size, image_size),
                       interpolation=cv2.INTER_NEAREST_EXACT)
    if preprocess_fn is not None:
        image = preprocess_fn(image)
        # rescale must be False when using preprocess_fn, otherwise
        # the results would be very poor
        rescale = False
    if rescale:
        # NOTE: segmentation needs rescale, classif model does not need
        # because they have their own preprocess_fn
        image = image.astype(np.float32) / 255.0
    return image


def get_all_keras_custom_objects() -> Dict[str, Callable]:
    """Get all the Keras model's custom_objects currently used in our application."""
    custom_objects = {"hybrid_loss": hybrid_loss, "focal_tversky": focal_tversky,
                      "iou_seg": iou_seg, 'Snake': Snake, 'GELU': GELU}
    return custom_objects


def get_segmentation_model_custom_objects(training_param: Dict[str, Any]) -> Dict[str, Callable]:
    """Get the custom objects required for loading the Keras segmentation model.

    NOTE: these custom metrics might change depending on how you built the model
     in `build_segmentation_model()`. Currently should only be these custom metrics:

    metrics = [hybrid_loss, iou_seg, focal_tversky]
    """
    use_hybrid_loss: bool = training_param['use_hybrid_loss']
    activation: str = training_param['activation']
    output_activation: str = training_param['output_activation']

    # KIV: For now, the metric names are taken from training_model instance and the metric
    # functions are dynamically generated using `eval`, so be sure to import the function
    # names directly on the script that uses this function.
    # metrics = list(
    #     session_state.new_training.training_model.metrics.keys())
    # custom_objects = {}
    # for m in metrics:
    #     if m != 'loss' and not m.startswith('val_') and m != 'categorical_crossentropy':
    #         custom_objects[m] = eval(m)
    #     elif m == 'loss':
    #         if use_hybrid_loss:
    #             custom_objects['hybrid_loss'] = hybrid_loss
    #         else:
    #             custom_objects['focal_tversky'] = focal_tversky

    # NOTE: putting hybrid_loss as the first one just in case
    custom_objects = {"hybrid_loss": hybrid_loss, "focal_tversky": focal_tversky,
                      "iou_seg": iou_seg}
    if not use_hybrid_loss:
        del custom_objects["hybrid_loss"]
    if activation == 'GELU':
        custom_objects['GELU'] = GELU
    if activation == 'Snake' or output_activation == 'Snake':
        custom_objects['Snake'] = Snake
    return custom_objects


def check_and_rename_folder_paths_for_backward_compatibility(
        X_test: List[str], y_test: List[str],
        pickle_path: Path, encoded_label_dict: Dict[int, str] = None):
    app_media_idx = X_test[0].find("app_media")
    base_data_dir = str(BASE_DATA_DIR)

    if _OLD_DIR_APP_NAME in X_test[0]:
        for i in range(len(X_test)):
            # replace the old data path with the new path
            X_test[i] = os.path.join(base_data_dir, X_test[i][app_media_idx:])

        with open(pickle_path, 'wb') as f:
            logger.info(
                "For backward compatibility of renaming "
                f"{_OLD_DIR_APP_NAME} to {_DIR_APP_NAME}, "
                "rewriting the pickle file with the renamed test set data paths."
                f"in {pickle_path}")

            if encoded_label_dict is not None:
                images_and_labels = (X_test, y_test, encoded_label_dict)
            else:
                images_and_labels = (X_test, y_test)
            pickle.dump(images_and_labels, f)


# NOTE: Clear cache cannot clear st.experimental_memo yet
# https://github.com/streamlit/streamlit/issues/3986
# @st.experimental_memo
@st.cache
def get_test_images_labels(
    pickle_path: Path,
    deployment_type: str) -> Union[Tuple[List[str], List[str], Dict[int, str]],
                                   Tuple[List[str], List[str]]]:
    logger.debug("Loading test set data from pickle file")
    with st.spinner("Getting test set images and labels ..."):
        with open(pickle_path, 'rb') as f:
            test_set_data = pickle.load(f)
        if deployment_type == 'Image Classification':
            X_test, y_test, encoded_label_dict = test_set_data
            check_and_rename_folder_paths_for_backward_compatibility(
                X_test, y_test, pickle_path,
                encoded_label_dict=encoded_label_dict
            )
            return X_test, y_test, encoded_label_dict
        elif deployment_type == 'Semantic Segmentation with Polygons':
            X_test, y_test = test_set_data
            check_and_rename_folder_paths_for_backward_compatibility(
                X_test, y_test, pickle_path
            )
            return X_test, y_test


# @st.cache(allow_output_mutation=True, show_spinner=False)
# @st.experimental_memo
def load_keras_model(model_path: Union[str, Path], metrics: List[Callable],
                     training_param: Dict[str, Any] = None):
    """Load the exported keras h5 model instead of only weights.

    The `custom_objects` is dynamically extracted from `training_model` instance
    for the segmentation model's custom loss functions or metrics.

    `metrics` should be obtained from Training.get_training_metrics().

    `training_param` is required for Semantic Segmentation with Polygons.

    Returns the Keras model instance."""
    if session_state.project.deployment_type == 'Semantic Segmentation with Polygons':
        assert training_param is not None
        custom_objects = get_segmentation_model_custom_objects(training_param)
    else:
        custom_objects = None
    tf.keras.backend.clear_session()
    gc.collect()
    model = tf.keras.models.load_model(model_path, custom_objects)
    # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=metrics)
    return model


def load_trained_keras_model(path: str):
    """To load user-uploaded model or trained project model (from other projects)"""
    tf.keras.backend.clear_session()
    gc.collect()
    all_custom_objects = get_all_keras_custom_objects()
    # load with all the custom objects used in our app
    # will raise ValueError if unknown custom_object is found in the model
    model = tf.keras.models.load_model(path, all_custom_objects)
    return model


def modify_trained_model_layers(model: tf.keras.Model, deployment_type: str,
                                input_shape: Tuple[int, int, int], num_classes: int,
                                compile: bool = False,
                                metrics: List[Callable] = None):
    """Modify the layers of the uploaded Keras model to have different input shape 
    and output shape. Input shape depends on the image_size or input_size of the 
    training_param, while the output shape depends on the number of classes (`num_classes`).
    """
    # NOTE: do not clear_session() when modifying the model midway here
    # https://newbedev.com/keras-replacing-input-layer
    model_config = model.get_config()
    # change the input shape
    model_config['layers'][0]['config']['batch_input_shape'] = (
        None, *(input_shape))

    all_custom_objects = get_all_keras_custom_objects()
    # if the model requires any unknown custom_objects, a ValueError will be raised
    new_model = model.__class__.from_config(model_config,
                                            custom_objects=all_custom_objects)

    # iterate over all the layers that we want to get weights from
    weights = [layer.get_weights() for layer in model.layers]
    for layer, weight in zip(new_model.layers, weights):
        layer.set_weights(weight)

    # ** Change the output layer
    # last layer's name is reused to replace the final layer to ensure unique names
    final_layer_name = new_model.layers[-1].name
    if deployment_type == "Image Classification":
        # use softmax for our training pipeline
        new_final_layer = tf.keras.layers.Dense(
            num_classes, activation='softmax',
            name=final_layer_name)
    elif deployment_type == "Semantic Segmentation with Polygons":
        # must use 'same' padding and 'softmax' activation
        new_final_layer = tf.keras.layers.Conv2D(
            num_classes, 1, padding='same',
            activation='softmax',
            name=final_layer_name)
    new_output = new_final_layer(new_model.layers[-2].output)
    final_model = tf.keras.Model(
        inputs=new_model.input, outputs=new_output)

    if compile or metrics:
        final_model.compile(loss=model.loss,
                            optimizer=model.optimizer,
                            metrics=metrics)
    return final_model

# ******************************* TFOD funcs *******************************


def xml_to_df(path: str) -> pd.DataFrame:
    """
    If a path to XML file is passed in, parse it directly.
    If directory is passed in, iterates through all .xml files (generated by our custom Label Studio) 
    in a given directory and combines them in a single Pandas dataframe.
    NOTE: This function will not work for the original Label Studio, because they export
    Pascal VOC XML files without the image extensions in the <filename> tags.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    if isinstance(path, Path):
        path = str(path)

    xml_list = []

    if os.path.isfile(path):
        xml_files = [path]
    else:
        xml_files = glob.glob(path + "/*.xml")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        for member in root.findall("object"):
            bndbox = member.find("bndbox")
            value = (
                filename,
                width,
                height,
                member.find("name").text,
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "classname",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def get_bbox_label_info(xml_df: pd.DataFrame,
                        image_name: str) -> Tuple[List[str], Tuple[int, int, int, int]]:
    """Get the class name and bounding box coordinates associated with the image.

    This is especially used to draw ground truth bounding boxes using `draw_gt_bboxes()`"""
    annot_df = xml_df.loc[xml_df['filename'] == image_name]
    class_names = annot_df['classname'].values
    bboxes = annot_df.loc[:, 'xmin': 'ymax'].values
    return class_names, bboxes


def generate_tfod_xml_csv(image_paths: List[str],
                          xml_dir: Path,
                          output_img_dir: Path,
                          csv_path: Path,
                          train_size: int,
                          transform: A.Compose):
    """Generate TFOD's CSV file for augmented images and bounding boxes used for generating TF Records.
    Also save the transformed images to the `output_img_dir` at the same time.

    `transform` is obtained from `get_transform()`.
    """

    output_img_dir.mkdir(parents=True, exist_ok=True)

    xml_df = xml_to_df(str(xml_dir))

    if train_size > len(image_paths):
        # randomly select the remaining paths and extend them to the original List
        # to make sure to go through the entire dataset for at least once
        n_remaining = train_size - len(image_paths)
        image_paths.extend(np.random.choice(
            image_paths, size=n_remaining, replace=True))

    logger.info('Generating CSV file for augmented bounding boxes ...')
    start = perf_counter()
    xml_list = []
    all_img_names = set()
    for image_path in stqdm(image_paths):
        image = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        class_names, bboxes = get_bbox_label_info(xml_df, filename)
        width, height = xml_df.loc[xml_df['filename'] == filename,
                                   'width': 'height'].values[0]

        transformed = transform(image=image, bboxes=bboxes,
                                class_names=class_names)
        transformed_image = transformed['image']

        # also save the transformed image at the same time to avoid doing it again later
        new_img_name = generate_image_name(filename, all_img_names)
        cv2.imwrite(str(output_img_dir / new_img_name), transformed_image)

        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.int32)
        # this 'class_names' key is based on the 'label_fields' in A.BboxParams()
        # used in get_transform()
        transformed_class_names = transformed['class_names']

        for bbox, class_name in zip(transformed_bboxes, transformed_class_names):
            value = (new_img_name, width, height, class_name, *bbox)
            xml_list.append(value)

    col_names = ["filename", "width", "height", "classname",
                 "xmin", "ymin", "xmax", "ymax"]

    xml_df = pd.DataFrame(xml_list, columns=col_names)
    xml_df.to_csv(csv_path, index=False)
    time_elapsed = perf_counter() - start
    logger.info(f"Done. {time_elapsed = :.4f} seconds")


def load_labelmap(labelmap_path):
    """
    Returns:
    category_index =
    {
        1: {'id': 1, 'name': 'category_1'},
        2: {'id': 2, 'name': 'category_2'},
        3: {'id': 3, 'name': 'category_3'},
        ...
    }
    """
    category_index = label_map_util.create_category_index_from_labelmap(
        labelmap_path,
        use_display_name=True)
    return category_index


def get_label_dict_from_labelmap(labelmap_path) -> Dict[int, str]:
    """Get encoded_label_dict for image classification/segmentation class names"""
    category_index = load_labelmap(labelmap_path)
    encoded_label_dict = {
        i: d['name'] for i, d in enumerate(category_index.values())
    }
    return encoded_label_dict


def get_ckpt_cnt(path: str):
    """Get the checkpoint number from the path (str, not Path)"""
    ckpt = path.split("ckpt-")[-1].split(".")[0]
    return int(ckpt)


def get_tfod_last_ckpt_path(ckpt_dir: Path) -> Path:
    """Find and return the latest TFOD checkpoint path. 

    The `ckpt_dir` should be `training_path['models']`.

    Return None if no ckpt-*.index file found"""
    ckpt_filepaths = glob.glob(str(ckpt_dir / 'ckpt-*.index'))
    if not ckpt_filepaths:
        logger.warning(
            f"There is no checkpoint file found in {ckpt_dir}, "
            "the TFOD model is not trained yet.")
        return None

    latest_ckpt = sorted(ckpt_filepaths, key=get_ckpt_cnt, reverse=True)[0]
    return Path(latest_ckpt)


# @st.cache(allow_output_mutation=True, show_spinner=False)
# @st.experimental_memo
def load_tfod_checkpoint(
        ckpt_dir: Path,
        pipeline_config_path: Path) -> Callable[[tf.Tensor], Dict[str, Any]]:
    """
    Loading from checkpoint instead of the exported savedmodel.

    The `ckpt_dir` should be `training_path['models']`.

    `pipeline_config_path` should be training_path['models'] / 'pipeline.config'
    """
    tf.keras.backend.clear_session()
    gc.collect()
    ckpt_path = get_tfod_last_ckpt_path(ckpt_dir)

    logger.info(f'Loading TFOD checkpoint from {ckpt_path} ...')
    start_time = perf_counter()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # need to remove the .index extension at the end
    ckpt.restore(str(ckpt_path).strip('.index')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    end_time = perf_counter()
    logger.info(f'Done! Took {end_time - start_time:.2f} seconds')
    return detect_fn


# @st.cache(allow_output_mutation=True, show_spinner=False)
# @st.experimental_memo
def load_tfod_model(saved_model_path: Path) -> Callable[[tf.Tensor], Dict[str, Any]]:
    """
    `saved_model_path` should be `training_path['export'] / 'saved_model'`

    NOTE: Caching is used on this method to avoid long loading times.
    Due to this, this method should not be used outside of training/deployment page.
    Maybe can improve this by using st.experimental_memo or other methods. Not sure.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    logger.info(f'Loading model from {saved_model_path} ...')
    start_time = perf_counter()
    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(str(saved_model_path))
    end_time = perf_counter()
    logger.info(
        f'Done Loading TFOD model! Took {end_time - start_time:.2f} seconds')
    return detect_fn


def tfod_detect(detect_fn: Callable[[tf.Tensor], Dict[str, Any]],
                image_np: np.ndarray,
                tensor_dtype=tf.uint8) -> Dict[str, Any]:
    """
    `detect_fn` is obtained using `load_tfod_model` or `load_tfod_checkpoint` functions. 
    `tensor_dtype` should be `tf.uint8` for exported model; 
    and `tf.float32` for checkpoint model to work. 
    """
    # Running the infernce on the image specified in the  image path
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    # input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    # input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = tf.expand_dims(input_tensor, 0)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tensor_dtype)

    # running detection using the loaded model: detect_fn
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)
    return detections


def get_detection_classes(
        detections: Dict[str, Any], category_index: Dict[int, Any],
        is_checkpoint: bool = False):
    # logger.debug(f"{detections['detection_classes'] = }")
    if is_checkpoint:
        # need offset for checkpoint
        unique_classes = np.unique(
            detections['detection_classes'] + 1)
    else:
        unique_classes = np.unique(
            detections['detection_classes'])
    pred_classes = [category_index[c]['name']
                    for c in unique_classes]
    return pred_classes


@st.cache
def get_tfod_test_set_data(test_data_dir: Path, return_xml_df: bool = True):
    with st.spinner("Getting images and annotations ..."):
        # test_data_dir should be in (training_path['images'] / 'test')
        logger.debug(f"Test set image directory: {test_data_dir}")
        test_img_paths = sorted(list_images(test_data_dir))
        # get the ground truth bounding box data from XML files
        if return_xml_df:
            gt_xml_df = xml_to_df(str(test_data_dir))
            return test_img_paths, gt_xml_df
        else:
            return test_img_paths

# ******************************* TFOD funcs *******************************

# *********************** Classification model funcs ***********************


# Specific input shapes for NasNet models when using "imagenet" weights
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/nasnet/NASNetLarge
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/nasnet/NASNetMobile
NASNET_IMAGENET_INPUT_SHAPES: Dict[str, Tuple[int, int, int]] = {
    'NASNetLarge': (331, 331, 3),
    'NASNetMobile': (224, 224, 3)
}

# to get the architecture's module name to obtain preprocess_input func
# for image classification models
ARCHITECTURE2MODULE_NAME: Dict[str, str] = {
    "densenet121": "densenet",
    "densenet169": "densenet",
    "densenet201": "densenet",
    "efficientnetb0": "efficientnet",
    "efficientnetb1": "efficientnet",
    "efficientnetb2": "efficientnet",
    "efficientnetb3": "efficientnet",
    "efficientnetb4": "efficientnet",
    "efficientnetb5": "efficientnet",
    "efficientnetb6": "efficientnet",
    "efficientnetb7": "efficientnet",
    "inceptionresnetv2": "inception_resnet_v2",
    "inceptionv3": "inception_v3",
    "mobilenet": "mobilenet",
    "mobilenetv2": "mobilenet_v2",
    "mobilenetv3large": "mobilenet_v3",
    "mobilenetv3small": "mobilenet_v3",
    "nasnetlarge": "nasnet",
    "nasnetmobile": "nasnet",
    "resnet101": "resnet",
    "resnet101v2": "resnet_v2",
    "resnet152": "resnet",
    "resnet152v2": "resnet_v2",
    "resnet50": "resnet",
    "resnet50v2": "resnet_v2",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "xception": "xception"
}


def find_architecture_name(keras_model: tf.keras.Model):
    # for all our classification models built in our app,
    # the architecture name can be found from the second layer
    try:
        model_name = keras_model.layers[1].name
    except IndexError:
        logger.error("Model has only one layer, which is not possible...?")
        return
    return model_name


def get_classif_model_preprocess_func(
        arch_name: str = None, keras_model: tf.keras.Model = None) -> Callable:
    """Get the preprocess_input function for the Keras pretrained classification model

    e.g. Architecture name (or `attached_model_name`) = `arch_name` = `"ResNet50"`
    """
    if not arch_name:
        assert keras_model is not None, (
            "Need Keras model to find architecture name for non-pretrained model")
        arch_name = find_architecture_name(keras_model)
    preprocess_module = ARCHITECTURE2MODULE_NAME.get(arch_name.lower())
    if not preprocess_module:
        logger.warning(f'Could not obtain preprocess_input function for "{arch_name}". '
                       'Skipping preprocessing function')
        return lambda x: x

    preprocess_input = attrgetter(
        f'{preprocess_module}.preprocess_input')(tf.keras.applications)
    return preprocess_input


def tf_classification_preprocess_input(
        imagePath: str, label: int,
        image_size: int, preprocess_fn: Callable = None,
        rescale: bool = False):
    """Using the `preprocess_fn` function obtained from
    `get_classif_model_preprocess_func()`"""
    raw = tf.io.read_file(imagePath)
    # decode using "INTEGER_ACCURATE" to achieve identical results with OpenCV
    # https://towardsdatascience.com/image-read-and-resize-with-opencv-tensorflow-and-pil-3e0f29b992be
    image = tf.image.decode_jpeg(
        raw, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.image.resize(image, (image_size, image_size), method='nearest')
    if preprocess_fn:
        image = preprocess_fn(image)
        # don't rescale when using preprocess_fn
        rescale = False
    if rescale:
        image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, dtype=tf.int32)
    return image, label


def classification_predict(preprocessed_img: np.ndarray, model: Any,
                           return_proba: bool = True) -> Union[Tuple[int, float], int]:
    y_proba = model.predict(np.expand_dims(preprocessed_img, axis=0))
    y_pred = np.argmax(y_proba, axis=-1)[0]

    if return_proba:
        return y_pred, y_proba.ravel()[y_pred]
    return y_pred

# *********************** Classification model funcs ***********************

# ************************ Segmentation model funcs ************************


def get_mask_path_from_image_path(image_path: str, mask_dir: Path):
    fname_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    mask_image_name = f"{fname_no_ext}.png"
    mask_path = mask_dir / mask_image_name
    return mask_path


def load_mask_image(ori_image_name: str, mask_dir: Path) -> np.ndarray:
    """Given the `ori_image_name` (refers to the original non-mask image),
    parse the mask image filename and find and load it from the `mask_dir` folder."""
    mask_image_name = os.path.splitext(ori_image_name)[0] + ".png"
    mask_path = mask_dir / mask_image_name
    logger.info(f"{mask_path = }")
    # MUST read in GRAYSCALE format to accurately preserve all the pixel values
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def get_coco_classes(
    json_path: Union[str, Path],
    return_coco: bool = True) -> Union[Tuple[COCO, List[int], bool, List[str]],
                                       List[str]]:
    """Get COCO classnames from the COCO JSON file.

    If `return_coco` is True, will return a tuple: (coco, catIDs, classnames)
    """
    coco = COCO(json_path)
    catIDs = coco.getCatIds()
    categories = coco.loadCats(catIDs)
    logger.debug(f"{categories = }")

    classnames = [cat["name"] for cat in categories]
    # add a background class at index 0
    if 'background' not in classnames:
        classnames = ["background"] + classnames
        # this flag is required for generate_mask_images() to know
        # whether the original COCO JSON already has a background class
        added_background = True
    else:
        added_background = False
    logger.debug(f"{classnames = }")

    if return_coco:
        return coco, catIDs, added_background, classnames

    return classnames


def generate_mask_images(coco_json_path: Union[str, Path] = None,
                         output_dir: Path = None,
                         n_masks: int = None,
                         verbose: bool = False,
                         st_container=None):
    """Generate mask images based on a COCO JSON file and save at `output_dir`

    Args:
        coco_json_path (Union[str, Path], optional): Path to the COCO JSON file.
            If None, infer from project export path. Defaults to None.
        output_dir (Path, optional): Path to output mask images.
            If None, infer from project export path. Defaults to None.
        n_masks (int, optional): Maximum number of mask images to generate,
            this is currently only used for the augmentation demo. If None, just 
            generate all mask images for all images in COCO JSON. Defaults to None.
        verbose (bool, optional): If True, will log the mask image info to console.
            Defaults to False.
        st_container ([type], optional): For `stqdm` progress bar. Can optionally pass
            in `st.sidebar`. Defaults to None.
    """
    data_export_dir = session_state.project.get_export_path()
    if not coco_json_path:
        coco_json_path = data_export_dir / "result.json"
    if not output_dir:
        output_dir = data_export_dir / "masks"
    os.makedirs(output_dir, exist_ok=True)

    json_file = json.load(open(coco_json_path))
    img_dict_list = json_file["images"]
    if n_masks:
        # optionally only generate certain number of mask images,
        # currently using this for the augmentation demo at the config page
        img_dict_list = img_dict_list[:n_masks]
    total_masks = len(img_dict_list)

    coco, catIDs, added_background, classnames = get_coco_classes(
        coco_json_path)

    logger.debug(f"Generating {total_masks} mask images in: {output_dir}")
    for img_dict in stqdm(img_dict_list, total=total_masks,
                          desc='Generating mask images', st_container=st_container):
        filename = os.path.basename(img_dict["file_name"])

        annIds = coco.getAnnIds(
            imgIds=img_dict["id"], catIds=catIDs, iscrowd=None)
        anns = coco.loadAnns(annIds)

        mask = np.zeros((img_dict["height"], img_dict["width"]))
        classes_found = []
        for annot in anns:
            current_annot_id = annot["category_id"]
            if added_background:
                # considering the extra added 'background' class at index 0
                # that did not exist in the COCO JSON file
                current_annot_id += 1
            className = classnames[current_annot_id]
            classes_found.append(className)
            pixel_value = classnames.index(className)
            # the final mask contains the pixel values for each class
            mask = np.maximum(coco.annToMask(annot) * pixel_value, mask)

        # save the mask images in PNG format to preserve the exact pixel values
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        success = cv2.imwrite(mask_path, mask)

        if verbose:
            logger.debug(f"Generated mask image for {mask_filename}")
            logger.debug(
                f"{filename} | "
                f"Unique pixel values = {np.unique(mask)} | "
                f"Unique classes found = {set(classes_found)} | "
                f"Number of annotations = {len(classes_found)}"  # or len(anns)
            )


def hybrid_loss(y_true, y_pred):
    loss_focal = focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
    loss_iou = iou_seg(y_true, y_pred)

    return loss_focal + loss_iou


def preprocess_mask(mask: np.ndarray, image_size: int, num_classes: int) -> tf.Tensor:
    # using bilinear to follow tensorflow default
    mask = cv2.resize(mask, (image_size, image_size),
                      interpolation=cv2.INTER_LINEAR)

    # this is a very important step to one-hot encode the mask
    # based on the number of classes, and keep in mind that
    # this `num_classes` will be the same as the number of filters of the
    # final output Conv2D layer in the model
    mask = tf.one_hot(mask, depth=num_classes, dtype=tf.int32)
    return mask


def segmentation_read_and_preprocess(
        imagePath: bytes, maskPath: bytes,
        image_size: int, num_classes: int) -> Tuple[np.ndarray, tf.Tensor]:
    # must decode the paths because they are in bytes format in TF operations
    image = cv2.imread(imagePath.decode())
    image = preprocess_image(image, image_size, rescale=True)

    mask = cv2.imread(maskPath.decode(), cv2.IMREAD_GRAYSCALE)
    mask = preprocess_mask(mask, image_size, num_classes)
    return image, mask


def segmentation_predict(model: Any,
                         preprocessed_img: np.ndarray,
                         original_width: int,
                         original_height: int) -> np.ndarray:
    pred_mask = model.predict(np.expand_dims(preprocessed_img, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    # index 0 to take away the one and only extra batch dimension
    pred_mask = pred_mask[0].astype(np.uint8)
    # resize it to original size after making prediction
    # take note of the order of width and height
    # using bilinear to follow tensorflow default
    pred_mask = cv2.resize(pred_mask, (original_width, original_height),
                           interpolation=cv2.INTER_LINEAR)
    return pred_mask

# ************************ Segmentation model funcs ************************
