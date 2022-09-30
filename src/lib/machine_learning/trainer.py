"""
Title: Training Management
Date: 01/10/2021
Author: Anson Tan Chen Tung
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================
"""
from __future__ import annotations

from functools import partial
from itertools import cycle
import json
import math
import os
import sys
import shutil
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING
import cv2
import numpy as np
import pickle
import glob
import tarfile
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wget
from sklearn.utils import shuffle
import streamlit as st
from streamlit import session_state

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from keras_unet_collection import models
from keras_unet_collection.losses import focal_tversky

from sklearn.metrics import classification_report, confusion_matrix
from imutils.paths import list_images

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from core.utils.file_handler import create_tarfile
if TYPE_CHECKING:
    from project.project_management import Project
    from training.training_management import Training, AugmentationConfig
from training.labelmap_management import Framework, Labels
from path_desc import DATASET_DIR, TFOD_DIR, PRE_TRAINED_MODEL_DIR, TFOD_MODELS_TABLE_PATH
from .command_utils import (export_tfod_savedmodel, find_tfod_eval_metrics, run_command,
                            run_command_update_metrics, find_tfod_metric)
from .utils import (NASNET_IMAGENET_INPUT_SHAPES, check_unique_label_counts, classification_predict, copy_images,
                    custom_train_test_split, find_architecture_name, generate_tfod_xml_csv, get_bbox_label_info, get_ckpt_cnt,
                    get_classif_model_preprocess_func, get_detection_classes, get_mask_path_from_image_path, get_test_images_labels,
                    get_tfod_last_ckpt_path, get_tfod_test_set_data, get_transform,
                    load_image_into_numpy_array, load_keras_model, load_labelmap,
                    load_tfod_checkpoint, load_tfod_model, load_trained_keras_model, modify_trained_model_layers, preprocess_image,
                    segmentation_predict, segmentation_read_and_preprocess,
                    tf_classification_preprocess_input, tfod_detect, hybrid_loss)
from .visuals import (PrettyMetricPrinter, create_class_colors, create_color_legend, draw_gt_bboxes,
                      draw_tfod_bboxes, get_colored_mask_image)
from .callbacks import LRTensorBoard, StreamlitOutputCallback
from deployment.utils import classification_inference_pipeline, tfod_inference_pipeline, segment_inference_pipeline


def create_labelmap_file(class_names: List[str], output_dir: Path, deployment_type: str):
    """`output_dir` is the directory to store the `labelmap.pbtxt` file"""
    labelmap_string = Labels.generate_labelmap_string(
        class_names,
        framework=Framework.TensorFlow,
        deployment_type=deployment_type)
    Labels.generate_labelmap_file(
        labelmap_string=labelmap_string,
        dst=output_dir,
        framework=Framework.TensorFlow,
        deployment_type=deployment_type)


class Trainer:
    def __init__(self, project: Project, new_training: Training):
        """
        You may choose to inherit this class if you would like to change the __init__ method,
        but make sure to retain all the attributes here for the other methods to work.
        """
        self.project_id: int = project.id
        self.training_id: int = new_training.id
        self.training_model_id: int = new_training.training_model.id
        self.attached_model_name: str = new_training.attached_model.name
        self.is_not_pretrained: bool = new_training.attached_model.is_not_pretrained
        self.training_model_name: str = new_training.training_model.name
        self.project_path: Path = project.get_project_path(project.name)
        # only used for segmentation to get the paths to mask images
        self.project_json_path: Path = project.get_project_json_path()
        self.deployment_type: str = project.deployment_type
        # with keys: 'train', 'eval', 'test'
        self.partition_ratio: Dict[str, float] = new_training.partition_ratio
        # if user selected both validation and testing partition ratio, we will have validation set
        self.has_valid_set = True if self.partition_ratio['test'] > 0 else False
        self.dataset_export_path: Path = project.get_export_path()
        self.training_param: Dict[str, Any] = new_training.training_param_dict
        self.class_names: List[str] = project.get_existing_unique_labels(
            for_training_id=self.training_id)
        if self.deployment_type == 'Semantic Segmentation with Polygons':
            # NOTE: must do it this way instead of using `get_coco_classes()` because
            #  the user might not use all classes when annotating (although rare case)
            #  and it could cause error
            if 'background' not in self.class_names:
                self.class_names = ['background'] + self.class_names
            if not self.is_not_pretrained:
                # only using this for Pre-trained models selected in our models_page
                self.segm_model_param, self.segm_model_func = (
                    new_training.get_segmentation_model_params(
                        self.training_param, return_model_func=True
                    ))
        self.metrics, self.metric_names = new_training.get_training_metrics()
        self.augmentation_config: AugmentationConfig = new_training.augmentation_config
        self.training_path: Dict[str, Path] = new_training.get_paths()
        # preprocess function specifically for image classification, to obtain later
        # on evaluation in run_keras_eval()
        self.preprocess_fn: Callable = None

    def __repr__(self):
        return "<{klass} {attrs}>".format(
            klass=self.__class__.__name__,
            attrs=" ".join("{}={!r}".format(k, v)
                           for k, v in self.__dict__.items() if v),
        )

    def train(self, is_resume: bool = False, stdout_output: bool = False,
              train_one_batch: bool = False):
        logger.debug("Clearing all existing Streamlit cache")
        # clearing all cache in case there is something weird happen with the
        # st.experimental_memo or st.cache methods
        st.legacy_caching.clear_cache()

        tf.keras.backend.clear_session()
        gc.collect()

        if self.training_path['export'].exists():
            # remove the exported model first before training
            shutil.rmtree(self.training_path['export'])

        if not is_resume:
            logger.info(f"Start new training for Training {self.training_id}")
        else:
            logger.info(f"Continue training for Training {self.training_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            if not is_resume:
                self.reset_tfod_progress()
            self.run_tfod_training(stdout_output, is_resume=is_resume)
        else:
            if train_one_batch:
                logger.info("Test training on only one batch of data")
            if not is_resume:
                self.reset_keras_progress()
            if self.deployment_type == "Image Classification":
                classification = True
                segmentation = False
            else:
                classification = False
                segmentation = True
            # optionally also allow train with one batch of data for classification/segmentation
            #  to test whether the model is good enough to overfit only one batch
            self.run_keras_training(is_resume=is_resume,
                                    classification=classification,
                                    segmentation=segmentation,
                                    train_one_batch=train_one_batch)
        # clear models cache
        st.legacy_caching.clear_cache()

    def evaluate(self):
        logger.info(f"Start evaluation for Training {self.training_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            self.run_tfod_evaluation()
        else:
            self.run_keras_eval()

    def export_model(self, re_export: bool = True):
        """
        TFOD: Export model and create a tarfile. If `re_export` is True,
        will export again even though the exported files already exist.

        Classification/segmentation: Move to paths['export'] and create a tarfile
        """
        logger.info(f"Exporting model for Training ID {self.training_id}, "
                    f"Model ID {self.training_model_id}")
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            self.export_tfod_model(stdout_output=False, re_export=re_export)
        else:
            self.export_keras_model()

    def reset_tfod_progress(self):
        # reset the training progress
        training_progress = {'Step': 0, 'Checkpoint': 0}
        session_state.new_training.reset_training_progress(
            training_progress)

    def reset_keras_progress(self):
        # reset the training progress
        training_progress = {'Epoch': 0}
        session_state.new_training.reset_training_progress(
            training_progress)

    def tfod_update_progress_metrics(self, cmd_output: str) -> Dict[str, float]:
        # ! DEPRECATED, update progress in real time during training using `run_command_update_metrics`
        """
        This function updates only the latest metrics from the **entire** stdout,
        instead of only from a single line. So this does not work for continuously
        updating the database during training.
        Refer 'run_command_update_metrics' function for continuous updates.
        """
        step = find_tfod_metric("Step", cmd_output)
        local_loss = find_tfod_metric(
            "Loss/localization_loss", cmd_output)
        reg_loss = find_tfod_metric(
            "Loss/regularization_loss", cmd_output)
        total_loss = find_tfod_metric(
            "Loss/total_loss", cmd_output)
        learning_rate = find_tfod_metric(
            "learning_rate", cmd_output)
        metrics = {
            "localization_loss": float(local_loss),
            "regularization_loss": float(reg_loss),
            "total_loss": float(total_loss),
            "learning_rate": float(learning_rate)
        }
        session_state.new_training.update_progress(step=int(step))
        session_state.new_training.update_metrics(metrics)
        return metrics

    def check_model_exists(self) -> Dict[str, bool]:
        """Check whether model weights (aka checkpoint) and exported model tarfile exists.
        This is to check whether our model needs to be exported first before letting the
        user to download."""
        paths = self.training_path
        ckpt_found = model_tarfile_found = exported_found = False
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            ckpt_path = get_tfod_last_ckpt_path(paths['models'])
            exported_model_path = paths['export'] / \
                'saved_model' / 'saved_model.pb'
            if exported_model_path.exists():
                exported_found = True
            else:
                logger.debug(f"{exported_model_path = }")
                logger.info("Object Detection Model is not exported yet.")
        else:
            ckpt_path = paths['output_keras_model_file']
            # Keras always has exported H5 model
            exported_found = True

        if ckpt_path and ckpt_path.exists():
            ckpt_found = True
        if paths['model_tarfile'].exists():
            model_tarfile_found = True

        return {'ckpt': ckpt_found,
                'exported_model': exported_found,
                'model_tarfile': model_tarfile_found}

    def run_tfod_training(self, stdout_output: bool = False, is_resume: bool = False):
        """
        Run training for TensorFlow Object Detection (TFOD) API.
        Can be used for Mask R-CNN model for image segmentation if wanted.
        Set `stdout_output` to True to print out the long console outputs
        generated from the script.
        """
        # training_param only consists of 'batch_size' and 'num_train_steps'

        # store this to update the total trained steps later when resuming training
        if is_resume:
            previous_trained_steps = session_state.new_training.progress['Step']

        # this name is used for the output model paths, see self.training_path
        CUSTOM_MODEL_NAME = self.training_model_name
        if self.is_not_pretrained:
            logger.info(f"Using user-uploaded model for TFOD with name: "
                        f"{self.attached_model_name}")
            uploaded_path = self.training_path['trained_model']
            logger.debug(f"User-uploaded model path: {uploaded_path}")
            # NOTE: these directories are structured so that the user-uploaded model
            #  files are within (pt_model_dir / pretrained_model_dirname)
            pt_model_dir = uploaded_path.parent
            pretrained_model_dirname = uploaded_path.name
        else:
            logger.info(f"Using pretrained model from TFOD with name: "
                        f"{self.attached_model_name}")
            # this df has columns: Model Name, Speed (ms), COCO mAP, Outputs, model_links
            models_df = pd.read_csv(TFOD_MODELS_TABLE_PATH, usecols=[
                                    'Model Name', 'model_links'])
            PRETRAINED_MODEL_URL = models_df.loc[
                models_df['Model Name'] == self.attached_model_name,
                'model_links'].squeeze()
            # this pretrained_model_dirname is different from self.attached_model_name,
            #  pretrained_model_dirname is the the first folder's name in the downloaded tarfile
            pretrained_model_dirname = PRETRAINED_MODEL_URL.split(
                "/")[-1].split(".tar.gz")[0]
            pt_model_dir = PRE_TRAINED_MODEL_DIR

        paths = self.training_path

        # ************* Generate train & test images in the folder *************
        with st.spinner('Generating train test splits ...'):
            logger.info('Generating train test splits')
            image_paths = sorted(list_images(
                self.dataset_export_path / "images"))
            labels = sorted(glob.glob(
                str(self.dataset_export_path / "Annotations" / "*.xml")))
            # for now we only makes use of test set for TFOD to make things simpler
            test_size = self.partition_ratio['eval'] + \
                self.partition_ratio['test']
            X_train, X_test, y_train, y_test = custom_train_test_split(
                # - BEWARE that the directories might be different if it's user uploaded
                image_paths=image_paths,
                test_size=test_size,
                labels=labels,
                no_validation=True
            )

        col, _ = st.columns([1, 1])
        with col:
            if self.augmentation_config.train_size is not None:
                # only train set is affected by augmentation, because we are generating
                # our own augmented images before training for TFOD
                train_size = self.augmentation_config.train_size
            else:
                train_size = len(y_train)
            st.info("TensorFlow Object Detection only uses a dedicated testing set "
                    "without validation set.")
            st.code(f"Total training images = {train_size}  \n"
                    f"Total testing images = {len(y_test)}")

        # initialize to check whether these paths exist for generating TF Records
        train_xml_csv_path = None
        with st.spinner('Copying images to folder, this may take awhile ...'):
            logger.info('Copying images to train test folder')
            if self.augmentation_config.exists():
                with st.spinner("Generating augmented training images ..."):
                    # get the transform from augmentation_config
                    transform = get_transform(self.augmentation_config,
                                              self.deployment_type)

                    # these csv files are temporarily generated to use for generating TF Records, should be removed later
                    train_xml_csv_path = paths['annotations'] / 'train.csv'
                    generate_tfod_xml_csv(
                        image_paths=X_train,
                        xml_dir=self.dataset_export_path / "Annotations",
                        output_img_dir=paths["images"] / 'train',
                        csv_path=train_xml_csv_path,
                        train_size=self.augmentation_config.train_size,
                        transform=transform
                    )
            else:
                # if not augmenting data, directly copy the train images to the folder
                copy_images(X_train,
                            dest_dir=paths["images"] / "train",
                            label_paths=y_train)
            # test set images should not be augmented
            copy_images(X_test,
                        dest_dir=paths["images"] / "test",
                        label_paths=y_test)
            logger.info("Dataset files copied successfully.")

        # ************* get the pretrained model if not exists *************
        downloaded_model_dir = pt_model_dir / pretrained_model_dirname
        if not self.is_not_pretrained and not downloaded_model_dir.exists():
            with st.spinner('Downloading pretrained model ...'):
                logger.info('Downloading pretrained model')
                wget.download(PRETRAINED_MODEL_URL, str(pt_model_dir))
                pretrained_tarfile = downloaded_model_dir.with_suffix(
                    '.tar.gz')
                with tarfile.open(pretrained_tarfile) as tar:
                    tar.extractall(pt_model_dir)
                    members = tar.getmembers()
                    # use the correct folder name, which could be different
                    folder_name = next(
                        m.name for m in members if not m.isfile())
                os.remove(pretrained_tarfile)
            downloaded_model_dir = downloaded_model_dir.with_name(folder_name)
            logger.info(f'Downloaded model at {downloaded_model_dir}')

        ckpt_file_stem = get_tfod_last_ckpt_path(
            downloaded_model_dir / 'checkpoint').stem

        # ****************** Create label_map.pbtxt ******************
        with st.spinner('Creating labelmap file ...'):
            CLASS_NAMES = self.class_names
            # must always generate a new one, regardless of which type of model
            logger.info(f"Creating labelmap file")
            create_labelmap_file(
                CLASS_NAMES, paths["labelmap_file"].parent, self.deployment_type)

        # ******************** Generate TFRecords ********************
        # this generate_tfrecord_st.py script is modified from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
        # to convert our PascalVOC XML annotation files into TFRecords for TFOD API
        GENERATE_TFRECORD_SCRIPT = LIB_PATH / \
            "machine_learning" / "module" / "generate_tfrecord_st.py"

        with st.spinner('Generating TFRecords ...'):
            logger.info('Generating TFRecords')
            image_extensions = 'jpeg jpg png'
            if train_xml_csv_path is not None:
                # using the CSV file generated during the augmentation process above
                # Must provide both image_dir and csv_path to skip the `xml_to_df` conversion step
                run_command(
                    f'python "{GENERATE_TFRECORD_SCRIPT}" '
                    f'-e {image_extensions} '
                    f'-i "{paths["images"] / "train"}" '
                    f'-l "{paths["labelmap_file"]}" '
                    f'-d "{train_xml_csv_path}" '
                    f'-o "{paths["annotations"] / "train.record"}"'
                )
            else:
                run_command(
                    f'python "{GENERATE_TFRECORD_SCRIPT}" '
                    f'-e {image_extensions} '
                    f'-x "{paths["images"] / "train"}" '
                    f'-l "{paths["labelmap_file"]}" '
                    f'-o "{paths["annotations"] / "train.record"}"'
                )
            # test set images are not augmented
            run_command(
                f'python "{GENERATE_TFRECORD_SCRIPT}" '
                f'-e {image_extensions} '
                f'-x "{paths["images"] / "test"}" '
                f'-l "{paths["labelmap_file"]}" '
                f'-o "{paths["annotations"] / "test.record"}"'
            )

        if not (paths["annotations"] / "train.record").exists() or \
                not (paths["annotations"] / "test.record").exists():
            txt = "Error generating TFRecords, please try again."
            logger.error(txt)
            st.error(txt)
            st.stop()

        logger.info("Successfully generated TFRecords")

        # ********************* pipeline.config *********************
        with st.spinner('Generating pipeline config file ...'):
            logger.info('Generating pipeline config file')
            original_config_path = downloaded_model_dir / 'pipeline.config'
            # copy over the pipeline.config file before modifying it
            shutil.copy2(original_config_path, paths["models"])

            # making pipeline.config file editable programmatically
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(paths["config_file"], "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

            pipeline_config = Labels.set_num_classes(
                len(CLASS_NAMES), pipeline_config=pipeline_config)

            logger.debug(f"{self.training_param = }")
            pipeline_config.train_config.batch_size = self.training_param['batch_size']
            num_train_steps = self.training_param['num_train_steps']
            if is_resume:
                # TFOD takes into account the total steps, instead of resume training
                # with the steps specified. But currently we allow the user to input
                # the num_train_steps the user wishes to resume training, so we need this
                num_train_steps += session_state.new_training.progress['Step']
            pipeline_config.train_config.num_steps = num_train_steps

            # To configure learning rate
            if self.training_param.get('is_changing_default_lr', False):
                opt_name = Labels.find_optimizer_field(pipeline_config)
                opt_field = getattr(
                    pipeline_config.train_config.optimizer, opt_name)
                lr_field = opt_field.learning_rate.cosine_decay_learning_rate
                lr_field.learning_rate_base = self.training_param[
                    'learning_rate_base']
                lr_field.total_steps = num_train_steps
                lr_field.warmup_learning_rate = self.training_param[
                    'warmup_learning_rate']
                lr_field.warmup_steps = self.training_param['warmup_steps']
                logger.debug(opt_field)

            # To configure paths
            pipeline_config.train_config.fine_tune_checkpoint = str(
                downloaded_model_dir / 'checkpoint' / ckpt_file_stem)
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path = str(
                paths["labelmap_file"])
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
                str(paths['annotations'] / 'train.record')]
            pipeline_config.eval_input_reader[0].label_map_path = str(
                paths["labelmap_file"])
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
                str(paths['annotations'] / 'test.record')]

            config_text = text_format.MessageToString(pipeline_config)
            with tf.io.gfile.GFile(paths["config_file"], "wb") as f:
                f.write(config_text)

            if not paths["config_file"].exists():
                txt = 'Unable to generate pipeline config file'
                st.error(txt)
                logger.error(txt)
                st.stop()

        logger.info(
            f'Successfully updated pipeline config file at {paths["config_file"]}')

        # ************************ TRAINING ************************
        # the path to the training script file `model_main_tf2.py` used to train our model

        TRAINING_SCRIPT = TFOD_DIR / \
            'research' / 'object_detection' / 'model_main_tf2.py'
        # NOTE: also save a checkpoint every 100 steps. And by default, only the latest
        # 7 checkpoints are kept. For parameter details, check the training script file's
        # model_lib_v2.train_loop(). SIDE NOTE: THIS is required to properly continue
        # training from the latest checkpoint! But checkpoint files could take up a lot of
        # space so be careful...
        checkpoint_every_n = 100

        start = perf_counter()
        command = (f'python "{TRAINING_SCRIPT}" '
                   f'--model_dir "{paths["models"]}" '
                   f'--pipeline_config_path "{paths["config_file"]}" '
                   f'--num_train_steps {num_train_steps} '
                   f'--checkpoint_every_n {checkpoint_every_n}')
        with st.spinner("**Training started ... This might take awhile ... "
                        "Do not refresh the page **"):
            logger.info('Start training')
            # stdout_output = True
            logger.debug(f"{stdout_output = }")
            traceback = run_command_update_metrics(
                command, num_train_steps, stdout_output=stdout_output,
                step_name='Step')
        if traceback:
            st.error(
                "Some error occurred while training, it could be due to insufficient "
                "memory (`ResourceExhaustedError`). You can check your console output for "
                "the error Traceback message.")
            if 'ResourceExhaustedError' in traceback:
                st.error(
                    f"There is not enough memory to perform the training with the batch "
                    f"size of **{self.training_param['batch_size']}**. Please try lowering "
                    "the batch size and train again.")
            return
        if not get_tfod_last_ckpt_path(paths['models']):
            st.error("Unknown error occurred while training, please check the terminal "
                     "output, or contact the admin.")
            logger.error("Error occurred while training")
            return
        time_elapsed = perf_counter() - start
        m, s = divmod(time_elapsed, 60)
        m, s = int(m), int(s)
        logger.info(f'Finished training! Took {m}m {s}s')
        st.success(f'Model has finished training! Took **{m}m {s}s**')

        # update checkpoint number only if training is successful
        progress = session_state.new_training.progress
        # progress['Checkpoint'] += 1
        latest_ckpt_path = get_tfod_last_ckpt_path(paths['models'])
        ckpt_cnt = get_ckpt_cnt(str(latest_ckpt_path))
        progress['Checkpoint'] = ckpt_cnt
        # if is_resume:
        #     # get the steps by adding to the trained steps of the previous session
        #     progress['Step'] += previous_trained_steps
        session_state.new_training.update_progress(
            progress, verbose=True)

        # ************************ EVALUATION ************************
        start = perf_counter()
        with st.spinner("Running object detection evaluation ..."):
            logger.info('Running object detection evaluation')
            command = (f'python "{TRAINING_SCRIPT}" '
                       f'--model_dir "{paths["models"]}" '
                       f'--pipeline_config_path "{paths["config_file"]}" '
                       f'--checkpoint_dir "{paths["models"]}"')
            filtered_outputs = run_command(
                command, stdout_output=stdout_output, st_output=False,
                filter_by=['DetectionBoxes_', 'Loss/'], is_cocoeval=True)
        if filtered_outputs:
            time_elapsed = perf_counter() - start
            m, s = divmod(time_elapsed, 60)
            m, s = int(m), int(s)
            logger.info(f'Finished COCO evaluation! Took {m}m {s}s')
            st.success(
                f'Model has finished COCO evaluation! Took **{m}m {s}s**')
            eval_result_text = find_tfod_eval_metrics(filtered_outputs)

            st.subheader("Object detection evaluation results on test set:")
            st.info(eval_result_text)

            # store the results to directly show on the training page in the future
            with open(paths['test_result_txt_file'], 'w') as f:
                f.write(eval_result_text)
            logger.debug("Saved evaluation script results at: "
                         f"{paths['test_result_txt_file']}")
        else:
            st.warning("There was some error occurred with COCO evaluation.")
            logger.error("There was some error occurred with COCO evaluation.")

        # Delete unwanted files excluding those needed for evaluation and exporting
        paths_to_del = (paths['annotations'],
                        paths["images"] / 'train', paths["export"])
        for p in paths_to_del:
            if p.exists():
                logger.debug("Removing unwanted directories used only "
                             f"for TFOD training: {p}")
                shutil.rmtree(p)

    def export_tfod_model(self, stdout_output: bool = False, re_export: bool = True):
        """ If `re_export` is True, will export again even though the exported
        files already exist."""
        paths = self.training_path

        export_tfod_savedmodel(paths, stdout_output, re_export)

        if hasattr(self, 'model'):
            logger.debug("Deleting loaded checkpoint model to be able to use "
                         "SavedModel instead")
            # delete the loaded checkpoint model if exists
            del self.model

        # copy the labelmap file into the export directory first
        # to store it for exporting
        shutil.copy2(paths['labelmap_file'], paths['export'])

        with st.spinner("Creating tarfile to download ..."):
            create_tarfile(paths['model_tarfile'].name,
                           target_path=paths['export'],
                           dest_dir=paths['model_tarfile'])

        logger.info("Congrats! Successfully created archived file at: "
                    f"{paths['model_tarfile']}")

    def run_tfod_evaluation(self):
        # get the required paths
        paths = self.training_path

        # show the stored evaluation result during training
        if paths['test_result_txt_file'].exists():
            st.subheader("Object Detection Evaluation results")
            with open(paths['test_result_txt_file']) as f:
                eval_result_text = f.read()
            st.info(eval_result_text)
        else:
            logger.error("Some error occurred while running COCO evaluation script, and "
                         f"the results were not saved at {paths['test_result_txt_file']}.")

        # **************** SHOW SOME IMAGES FOR EVALUATION ****************
        st.subheader("Prediction Results on Validation/Test Set:")

        exist_dict = self.check_model_exists()
        with st.spinner("Loading model ... This might take awhile ..."):
            # only use the full exported model after the user has exported it
            if exist_dict['exported_model']:
                if not hasattr(self, 'model'):
                    self.model = load_tfod_model(
                        self.training_path['export'] / 'saved_model')
                is_checkpoint = False
            else:
                if not hasattr(self, 'model'):
                    self.model = load_tfod_checkpoint(
                        ckpt_dir=self.training_path['models'],
                        pipeline_config_path=self.training_path['config_file'])
                is_checkpoint = True
        category_index = load_labelmap(paths['labelmap_file'])
        logger.debug(f"{category_index = }")

        options_col, _ = st.columns([1, 1])
        prev_btn_col_1, next_btn_col_1, _ = st.columns([1, 1, 3])
        true_img_col, pred_img_col = st.columns([1, 1])
        prev_btn_col_2, next_btn_col_2, _ = st.columns([1, 1, 3])

        if 'start_idx' not in session_state:
            # to keep track of the image index to show
            session_state['start_idx'] = 0

        # take the test set images
        test_data_dir = paths['images'] / 'test'
        image_paths, gt_xml_df = get_tfod_test_set_data(test_data_dir)

        with options_col:
            def reset_start_idx():
                session_state['start_idx'] = 0

            n_samples = st.number_input(
                "Number of samples to show:",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                format='%d',
                key='n_samples',
                help="Number of samples to run detection and display results.",
                on_change=reset_start_idx
            )
            conf_threshold = st.number_input(
                "Minimum confidence threshold:",
                min_value=0.1,
                max_value=0.99,
                value=0.6,
                step=0.01,
                format='%.2f',
                key='conf_threshold',
                help=("If a prediction's confidence score exceeds this threshold, "
                      "then it will be displayed, otherwise discarded."),
            )

            total_samples = len(image_paths)
            st.info(f"**Total test set images**: {total_samples}")
            if not exist_dict['exported_model']:
                with st.expander("NOTES about inference with non-exported model:"):
                    st.markdown("""You are using the model loaded from a
                    checkpoint, the inference time will be slightly slower
                    than an exported model.  You may use this to check the
                    performance of the model on real images before deciding
                    to export the model. Or you may choose to export the
                    model by clicking the "Export Model" button above first
                    (if this is in training page)
                    before checking the evaluation results. Note that the
                    inference time for the first image will take significantly
                    longer than others. In production, we will also export the
                    model before deploying it for inference.""")

        # must try to int them to avoid complications
        start_idx, n_samples = int(session_state['start_idx']), int(n_samples)
        current_image_paths = image_paths[start_idx: start_idx + n_samples]

        start = start_idx + 1
        end = start_idx + n_samples
        end = end if end <= total_samples else total_samples
        logger.info(f"Detecting from the test set images: {start}"
                    f" to {end} ...")
        options_col.info(
            f"Showing sample images: **{start}** to **{end}**")

        def previous_samples():
            if session_state['start_idx'] > 0:
                session_state['start_idx'] -= n_samples

        def next_samples():
            max_start = total_samples - n_samples
            if session_state['start_idx'] < max_start:
                session_state['start_idx'] += n_samples

        prev_btn_col_1.button('⏮️ Previous samples', key='btn_prev_images_1',
                              on_click=previous_samples)
        next_btn_col_1.button('Next samples ⏭️', key='btn_next_images_1',
                              on_click=next_samples)

        with st.spinner("Running detections ..."):
            # create the colors for each class to draw the bboxes nicely
            class_colors = create_class_colors(self.class_names)

            for i, p in enumerate(current_image_paths):
                # current_img_idx = start_idx + i + 1
                logger.debug(f"Detecting on image at: {p}")
                img = cv2.imread(str(p))
                filename = os.path.basename(p)

                start_t = perf_counter()
                results = tfod_inference_pipeline(
                    img, model=self.model,
                    conf_threshold=conf_threshold,
                    category_index=category_index,
                    is_checkpoint=is_checkpoint)
                detections = results['detections']
                img_with_detections = results['img']
                time_elapsed = perf_counter() - start_t
                logger.info(f"Done inference on {filename}. "
                            f"[{time_elapsed:.2f} secs]")

                pred_classes = get_detection_classes(
                    detections, category_index, is_checkpoint=is_checkpoint)
                logger.info(f"Detected classes: {pred_classes}")

                class_names, bboxes = get_bbox_label_info(gt_xml_df, filename)
                img = draw_gt_bboxes(img, bboxes,
                                     class_names=class_names,
                                     class_colors=class_colors)
                true_img_col.image(img, channels='BGR',
                                   caption=f'{start + i}. Ground Truth: {filename}')
                pred_img_col.image(img_with_detections, channels='RGB',
                                   caption=f'Prediction: {filename}')

        prev_btn_col_2.button('⏮️ Previous samples', key='btn_prev_images_2',
                              on_click=previous_samples)
        next_btn_col_2.button('Next samples ⏭️', key='btn_next_images_2',
                              on_click=next_samples)

    # ********************* METHODS FOR KERAS TRAINING *********************

    def get_preprocess_fn(self, keras_model: tf.keras.Model = None) -> Callable:
        # NOTE: this preprocess_input function makes a huge difference on
        # the model performance
        if self.is_not_pretrained:
            assert keras_model is not None, (
                "Need Keras model to find architecture name for non-pretrained model")
            architecture_name = find_architecture_name(keras_model)
        else:
            architecture_name = self.attached_model_name
        preprocess_fn = get_classif_model_preprocess_func(
            architecture_name)
        return preprocess_fn

    def create_tf_dataset(self, X_train: List[str], y_train: List[str],
                          X_test: List[str], y_test: List[str],
                          X_val: List[str] = None, y_val: List[str] = None,
                          keras_model: tf.keras.Model = None):
        """Create tf.data.Dataset for training set and testing set; also optionally
        create for validation set if passed in."""
        logger.debug(f"Creating TF dataset for {self.deployment_type}")
        image_size = self.training_param['image_size']
        if self.deployment_type == 'Image Classification':
            preprocess_fn = self.get_preprocess_fn(keras_model)

            def tf_preprocess_fn(image):
                # wrap the function and use it as a TF operation
                # to optimize for performance
                image = tf.numpy_function(
                    func=preprocess_fn, inp=[image], Tout=tf.float32)
                return image
            tf_preprocess_data = partial(tf_classification_preprocess_input,
                                         image_size=image_size,
                                         #  preprocess_fn=None)
                                         preprocess_fn=tf_preprocess_fn)
        else:
            num_classes = len(self.class_names)
            logger.debug(f"{num_classes = }")
            # preprocess_fn = tf_segmentation_preprocess_input
            preprocess_fn = partial(segmentation_read_and_preprocess,
                                    image_size=image_size,
                                    num_classes=num_classes)

            def tf_preprocess_data(imagePath: str, maskPath: str) -> Tuple[tf.Tensor, tf.Tensor]:
                # wrap the function and use it as a TF operation
                # to optimize for performance
                image, mask = tf.numpy_function(
                    preprocess_fn,
                    inp=[imagePath, maskPath],
                    Tout=[tf.float32, tf.int32]
                )
                return image, mask

        if self.augmentation_config.exists():
            # get the Albumentations transform
            transform = get_transform(self.augmentation_config,
                                      self.deployment_type)

            if self.deployment_type == 'Image Classification':
                # https://albumentations.ai/docs/examples/tensorflow-example/
                def aug_fn(image):
                    aug_data = transform(image=image)
                    aug_img = aug_data["image"]
                    return aug_img

                def augment(image, label):
                    aug_img = tf.numpy_function(
                        func=aug_fn, inp=[image], Tout=tf.float32)
                    return aug_img, label
            else:
                def aug_fn(image, mask):
                    aug_data = transform(image=image, mask=mask)
                    aug_img = aug_data["image"]
                    aug_mask = aug_data["mask"]
                    return aug_img, aug_mask

                def augment(image, mask):
                    aug_img, aug_mask = tf.numpy_function(
                        func=aug_fn, inp=[image, mask], Tout=[tf.float32, tf.int32])
                    return aug_img, aug_mask

        if self.deployment_type == 'Image Classification':
            def set_shapes(img, label, img_shape=(image_size, image_size, 3)):
                img.set_shape(img_shape)
                label.set_shape([])
                return img, label
        else:
            def set_shapes(img, mask):
                # set the shape to let TensorFlow knows about the shape
                # just like `assert` statements to ensure the shapes are correct
                # NOTE: this step is required to show metrics during training
                img.set_shape([image_size, image_size, 3])
                mask.set_shape([image_size, image_size, num_classes])
                return img, mask

        # randomly shuffle once here, then shuffle with smaller buffer size later
        # NOTE: must shuffle both image_paths and labels together!!
        X_train, y_train = shuffle(X_train, y_train)

        # only train set is augmented and shuffled
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = self.training_param['batch_size']
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # NOTE: large shuffle takes up too much memory and could be very slow
        # cache() also takes up too much memory on large dataset
        shuffle_size = len(X_train) if len(X_train) < 1000 else 1000
        train_ds = train_ds.map(
            tf_preprocess_data, num_parallel_calls=AUTOTUNE)
        if self.augmentation_config.exists():
            train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
        train_ds = (
            train_ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
            .shuffle(shuffle_size)
            # .cache()
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = (
            test_ds.map(tf_preprocess_data,
                        num_parallel_calls=AUTOTUNE)
            .map(set_shapes, num_parallel_calls=AUTOTUNE)
            # .cache()
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

        if X_val:
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_ds = (
                val_ds.map(tf_preprocess_data,
                           num_parallel_calls=AUTOTUNE)
                .map(set_shapes, num_parallel_calls=AUTOTUNE)
                # .cache()
                .batch(batch_size)
                .prefetch(AUTOTUNE)
            )
            return train_ds, val_ds, test_ds

        return train_ds, test_ds

    def build_classification_model(self):
        # run this only if you plan to completely rebuild the model
        # do not run this in the middle of building the model
        keras.backend.clear_session()
        # e.g. ResNet50
        model_name = self.attached_model_name
        image_size = self.training_param['image_size']
        input_shape = (image_size, image_size, 3)

        if 'nasnet' in model_name.lower():
            # nasnet only supports a specific input_shape for "imagenet" weights,
            # check NASNetLarge and NASNetMobile docs for details
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/nasnet/NASNetLarge
            required_input_shape = NASNET_IMAGENET_INPUT_SHAPES.get(model_name)
            logger.debug(f"{model_name = }, {required_input_shape = }")
            if input_shape == required_input_shape:
                weights = "imagenet"
            else:
                logger.info(f"Not using pretrained imagenet weights for '{model_name}' "
                            "due to the requirement of a specific input_shape: "
                            f"{required_input_shape}")
                weights = None
        else:
            weights = "imagenet"

        pretrained_model_func = getattr(tf.keras.applications,
                                        model_name)
        baseModel = pretrained_model_func(
            weights=weights, include_top=False,
            input_shape=input_shape
        )

        # freeze the pretrained model
        baseModel.trainable = False

        # referring https://www.tensorflow.org/guide/keras/transfer_learning
        inputs = keras.Input(shape=input_shape)
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, especially
        # for BatchNormalization layer, which acts differently in inference mode.
        headModel = baseModel(inputs, training=False)

        # then add extra layers to suit our choice
        # AveragePooling2D layer might need lower pool_size depending on the
        # the different input sizes due to different pretrained models and dataset size
        error = True
        pool_sizes = (7, 5)
        for pool_size in pool_sizes:
            try:
                pooled = AveragePooling2D(pool_size=pool_size)(headModel)
            except Exception as e:
                msg = e
                logger.debug(msg)
            else:
                error = False
                logger.info(
                    f"Using AveragePooling2D with pool_size = {pool_size}")
                headModel = pooled

                headModel = Flatten(name="flatten")(headModel)
                break
        if error:
            logger.info(
                "Skipping AveragePooling2D layer due to small layer's input_size")
            logger.info("Using GlobalAveragePooling2D layer")
            # GlobalAveragePooling2D does not need Flatten layer
            headModel = GlobalAveragePooling2D()(headModel)

        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        # the last layer is the most important to ensure the model outputs
        #  the result that we want
        headModel = Dense(
            len(self.class_names), activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        # model = keras.Model(inputs=baseModel.input, outputs=headModel)
        model = keras.Model(inputs=inputs, outputs=headModel)

        optimizer_func = getattr(tf.keras.optimizers,
                                 self.training_param['optimizer'])
        lr = self.training_param['learning_rate']

        # using a simple decay for now, can use cosine annealing if wanted
        opt = optimizer_func(learning_rate=lr,
                             decay=lr / self.training_param['num_epochs'])
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt, metrics=self.metrics)
        return model

    def build_segmentation_model(self):
        keras.backend.clear_session()

        model_param_dict = self.segm_model_param
        model = getattr(models, self.segm_model_func)(**model_param_dict)

        optimizer_func = getattr(tf.keras.optimizers,
                                 self.training_param['optimizer'])
        lr = self.training_param['learning_rate']

        # using a simple decay for now, can use cosine annealing if wanted
        opt = optimizer_func(learning_rate=lr,
                             decay=lr / self.training_param['num_epochs'])
        use_hybrid_loss = self.training_param['use_hybrid_loss']
        # focal_tversky seems to be good default
        # in the future, perhaps also allow use to choose a loss function
        loss = hybrid_loss if use_hybrid_loss else focal_tversky

        model.compile(loss=loss, optimizer=opt, metrics=self.metrics)
        return model

    def load_and_modify_trained_model(self):
        """
        This function is used to modify the trained model (uploaded or project model 
        from other projects) to use it to continue training with the current project 
        datasets.
        """
        assert self.is_not_pretrained
        logger.info("Loading the trained keras model")
        model = load_trained_keras_model(
            str(self.training_path['trained_keras_model_file']))

        image_size = self.training_param['image_size']
        input_shape = (image_size, image_size, 3)

        model = modify_trained_model_layers(
            model, self.deployment_type,
            input_shape=input_shape,
            num_classes=len(self.class_names),
            # must compile
            compile=True,
            metrics=self.metrics)
        return model

    def create_callbacks(self, train_size: int,
                         progress_placeholder: Dict[str, Any],
                         num_epochs: int,
                         update_metrics: bool = True) -> List[Callback]:
        # this callback saves the checkpoint with the best `val_loss`
        ckpt_cb = ModelCheckpoint(
            filepath=self.training_path['model_weights_file'],
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        # maybe early_stopping is not necessary
        # early_stopping_cb = EarlyStopping(patience=10, monitor='val_loss',
        #                                   mode='min', restore_best_weights=True)

        tensorboard_cb = LRTensorBoard(
            log_dir=self.training_path['tensorboard_logdir'])

        pretty_metric_printer = PrettyMetricPrinter()
        steps_per_epoch = math.ceil(
            train_size / self.training_param['batch_size'])
        st_output_cb = StreamlitOutputCallback(
            pretty_metric_printer,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            progress_placeholder=progress_placeholder,
            refresh_rate=20,
            update_metrics=update_metrics
        )

        return [ckpt_cb, tensorboard_cb, st_output_cb]

    def load_model_weights(self, model: keras.Model = None, build_model: bool = False):
        """Load the model weights for evaluation or inference. 
        If `build_model` is `True`, the model will be rebuilt and load back the weights.
        Or just send in a built model to directly use it to load the weights"""
        if build_model:
            if self.deployment_type == 'Image Classification':
                model = self.build_classification_model()
            else:
                model = self.build_segmentation_model()
        if model is not None:
            model.load_weights(self.training_path['model_weights_file'])
            return model
        else:
            st.error(
                "Model weights is not loaded. Some error has occurred. Please try again.")
            logger.error(f"""Model weights for {self.training_model_id} is not loaded,
            please pass in a `model` or set `build_model` to True to rebuild the model
            layers.""")
            st.stop()

    def run_keras_training(self,
                           is_resume: bool = False,
                           train_one_batch: bool = False,
                           classification: bool = False,
                           segmentation: bool = False):
        assert any((classification, segmentation)) \
            and not all((classification, segmentation)), (
            "Please choose only one of the task"
        )

        if classification:
            # getting the CSV file exported from the dataset
            # the CSV file generated has these columns:
            # image, id, label, annotator, annotation_id, created_at, updated_at, lead_time.
            # The first col `image` contains the relative paths to the images
            dataset_df = pd.read_csv(self.dataset_export_path / 'result.csv',
                                     dtype={'image': str, 'label': 'category'})
            # convert the relative paths to absolute paths
            dataset_df['image'] = dataset_df['image'].apply(
                lambda x: str(DATASET_DIR / x))
            image_paths = dataset_df['image'].tolist()
            label_series = dataset_df['label']
            # encoded labels, e.g. ['cat', 'dog'] -> [0, 1]
            labels = label_series.cat.codes.tolist()
            # use this to store the classnames with their indices as keys
            encoded_label_dict = dict(enumerate(label_series.cat.categories))
            logger.debug(f"{encoded_label_dict = }")
            ok = check_unique_label_counts(labels, encoded_label_dict)
            stratify = True if ok else False
            show_class_distribution = True
        elif segmentation:
            labelstudio_json = json.load(open(self.project_json_path))
            # using these paths instead to allow removing the 'images' folder later
            # while still retaining the paths to access the original dataset images
            image_paths = [str(DATASET_DIR / i['data']['image'])
                           for i in labelstudio_json]
            # image_dir = self.dataset_export_path / 'images'
            # image_paths = sorted(list_images(image_dir))
            mask_dir = self.dataset_export_path / 'masks'
            labels = [str(get_mask_path_from_image_path(
                p, mask_dir)) for p in image_paths]
            stratify = False
            # initialize this to pass to custom_train_test_split()
            encoded_label_dict = None
            show_class_distribution = False
            # coco_json_path = self.dataset_export_path / 'result.json'
            # json_file = json.load(open(coco_json_path))

        # ************* Generate train & test images in the folder *************
        if self.has_valid_set:
            with st.spinner('Generating train test splits ...'):
                X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(
                    # - BEWARE that the directories might be different if it's user uploaded
                    image_paths=image_paths,
                    test_size=self.partition_ratio['test'],
                    val_size=self.partition_ratio['eval'],
                    labels=labels,
                    no_validation=False,
                    stratify=stratify,
                    encoded_label_dict=encoded_label_dict,
                    show_class_distribution=show_class_distribution,
                )

            col, _ = st.columns([1, 1])
            with col:
                st.code(f"Total training images = {len(X_train)}  \n"
                        f"Total validation images = {len(X_val)}  \n"
                        f"Total testing images = {len(X_test)}")
        else:
            # the user did not select a test size, i.e. only using validation set for testing,
            # thus for our custom_train_test_split implementation, we assume we only
            # want to use the test_size without val_size (sorry this might sound confusing)
            with st.spinner('Generating train test splits ...'):
                X_train, X_test, y_train, y_test = custom_train_test_split(
                    # - BEWARE that the directories might be different if it's user uploaded
                    image_paths=image_paths,
                    test_size=self.partition_ratio['eval'],
                    labels=labels,
                    no_validation=True,
                    stratify=stratify,
                    encoded_label_dict=encoded_label_dict,
                    show_class_distribution=show_class_distribution,
                )

            col, _ = st.columns([1, 1])
            with col:
                st.info("""No test size was selected in the training config page.
                So validation set is used as the test set instead.""")
                st.code(f"Total training images = {len(X_train)}  \n"
                        f"Total validation images = {len(X_test)}")

        with st.spinner("Saving the test set data ..."):
            if classification:
                images_and_labels = (X_test, y_test, encoded_label_dict)
            else:
                images_and_labels = (X_test, y_test)
            with open(self.training_path['test_set_pkl_file'], "wb") as f:
                logger.debug("Dumping test set data as pickle file "
                             f"in {self.training_path['test_set_pkl_file']}")
                pickle.dump(images_and_labels, f)

        # *********************** Build model ***********************
        if not is_resume:
            model_name = self.attached_model_name
            with st.spinner(f"Building the model based on '{model_name}' architecture ..."):
                if self.is_not_pretrained:
                    logger.info("Loading model from uploaded or project model from "
                                "other projects")
                    model = self.load_and_modify_trained_model()
                else:
                    logger.info(
                        f"Building model based on '{model_name}' architecture")
                    if classification:
                        model = self.build_classification_model()
                    else:
                        model = self.build_segmentation_model()
        else:
            logger.info(f"Loading Model ID {self.training_model_id} "
                        "to resume training ...")
            with st.spinner("Loading trained model to resume training ..."):
                # load the full Keras model instead of weights to easily resume training
                logger.info(
                    f"Loading trained Keras model for {self.deployment_type}")
                model = load_keras_model(
                    self.training_path['output_keras_model_file'],
                    self.metrics, self.training_param)

        # ***************** Preparing tf.data.Dataset *****************
        # this comes after building model to be able to pass the model
        # into self.create_tf_dataset()
        with st.spinner("Creating TensorFlow dataset ..."):
            logger.info("Creating TensorFlow dataset")
            batch_size = self.training_param['batch_size']
            if self.has_valid_set:
                if train_one_batch:
                    # take only one batch for test run
                    X_train, y_train, X_test, y_test, X_val, y_val = (
                        X_train[:batch_size], y_train[:batch_size], X_test[:batch_size],
                        y_test[:batch_size], X_val[:batch_size], y_val[:batch_size]
                    )
                train_ds, val_ds, test_ds = self.create_tf_dataset(
                    X_train, y_train, X_test, y_test, X_val, y_val,
                    keras_model=model)
            else:
                if train_one_batch:
                    # take only one batch for test run
                    X_train, y_train, X_test, y_test = (
                        X_train[:batch_size], y_train[:batch_size],
                        X_test[:batch_size], y_test[:batch_size]
                    )
                train_ds, test_ds = self.create_tf_dataset(
                    X_train, y_train, X_test, y_test, keras_model=model)

        # ********* Preparing for training, including callbacks *********
        if not is_resume:
            initial_epoch = 0
            num_epochs = self.training_param['num_epochs']
        else:
            initial_epoch = session_state.new_training.progress['Epoch']
            logger.debug(f"{initial_epoch = }")
            num_epochs = initial_epoch + self.training_param['num_epochs']

        progress_placeholder = {}
        progress_placeholder['epoch'] = st.empty()
        progress_placeholder['batch'] = st.empty()
        # not updating progress & metrics if test training on one batch of data
        update_metrics = False if train_one_batch else True
        callbacks = self.create_callbacks(
            train_size=len(y_train), progress_placeholder=progress_placeholder,
            num_epochs=num_epochs, update_metrics=update_metrics)

        # ********************** Train the model **********************
        if self.has_valid_set:
            validation_data = val_ds
        else:
            validation_data = test_ds

        logger.info("Training model...")
        start = perf_counter()
        with st.spinner("Training model ..."):
            model.fit(
                train_ds,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                epochs=num_epochs,
                callbacks=callbacks,
                # turn off printing training progress in console
                verbose=0
            )
        time_elapsed = perf_counter() - start

        m, s = divmod(time_elapsed, 60)
        m, s = int(m), int(s)
        logger.info(f'Finished training! Took {m}m {s}s')
        st.success(f'Model has finished training! Took **{m}m {s}s**')

        with st.spinner("Loading the model with the best validation loss ..."):
            model = self.load_model_weights(model)

        if not train_one_batch:
            with st.spinner("Saving the trained TensorFlow model ..."):
                model.save(self.training_path['output_keras_model_file'],
                           save_traces=True)
                logger.debug(f"Keras model saved at "
                             f"{self.training_path['output_keras_model_file']}")

        # ************************ Evaluation ************************
        with st.spinner("Evaluating on validation set and test set if available ..."):
            # NOTE: the loss function might change depending on which one you chose
            test_output = model.evaluate(test_ds)
            test_txt = []

            if self.has_valid_set:
                val_output = model.evaluate(val_ds)
                val_txt = []
                for name, val_l, test_l in zip(self.metric_names, val_output,
                                               test_output):
                    val_txt.append(f"**Validation {name}**: {val_l:.4f}")
                    test_txt.append(f"**Testing {name}**: {test_l:.4f}")
                result_txt = "  \n".join(val_txt + test_txt)
            else:
                for name, test_l in zip(self.metric_names, test_output):
                    test_txt.append(f"**Validation {name}**: {test_l:.4f}")
                result_txt = "  \n".join(test_txt)

            # display the result info
            st.info(result_txt)

        if classification:
            # show a nicely formatted classification report
            with st.spinner("Making predictions on the test set ..."):
                pred_proba = model.predict(test_ds)
                preds = np.argmax(pred_proba, axis=-1)
                y_true = np.concatenate([y for _, y in test_ds], axis=0)
                unique_label_ids = np.unique((y_true, preds))
                target_names = [str(encoded_label_dict[i])
                                for i in unique_label_ids]
                logger.debug(f"{target_names = }")

            with st.spinner("Generating classification report ..."):
                classif_report = classification_report(
                    y_true, preds, target_names=target_names, zero_division=0
                )
                st.subheader("Classification report")
                st.text(classif_report)

                # append to the test results
                header_txt = "Classification report:"
                result_txt = "  \n".join(
                    [result_txt, header_txt, classif_report])

            with st.spinner("Creating confusion matrix ..."):
                cm = confusion_matrix(y_true, preds)

                fig = plt.figure()
                ax = sns.heatmap(
                    cm, cmap="Blues", annot=True, fmt="d", cbar=False,
                    yticklabels=target_names, xticklabels=target_names,
                )

                plt.title("Confusion Matrix", size=12, fontfamily="serif")
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                # save the figure to reuse later
                plt.savefig(str(self.training_path['confusion_matrix_file']),
                            bbox_inches="tight")
                st.pyplot(fig)

        # save the results in a txt file to easily show again later
        with open(self.training_path['test_result_txt_file'], "w") as f:
            f.write(result_txt)
        logger.debug(
            f"Test set result saved at {self.training_path['test_result_txt_file']}")

        # create a labelmap_file for the user to contain the class labels
        logger.info("Generating labelmap file to store class labels")
        create_labelmap_file(
            self.class_names,
            self.training_path["labelmap_file"].parent,
            self.deployment_type)

        # remove the model weights file, which is basically just used for loading
        # the best weights with the lowest val_loss. Decided to just use the
        # full model h5 file to make things easier to resume training.
        weights_path = self.training_path['model_weights_file']
        logger.debug(f"Removing unused model_weights file: {weights_path}")
        os.remove(weights_path)

        # this only exists for segmentation task for now
        exported_image_dir = self.dataset_export_path / 'images'
        if exported_image_dir.exists():
            logger.info("Removing unused exported training images")
            shutil.rmtree(self.dataset_export_path / 'images')

    def run_keras_eval(self):
        if self.training_path['test_result_txt_file'].exists():
            # show the evaluation results stored during training
            with open(self.training_path['test_result_txt_file']) as f:
                result_txt = f.read()
            st.subheader("Evaluation result on validation set and "
                         "test set if available:")
            result_col, _ = st.columns(2)
            with result_col:
                if self.deployment_type == 'Image Classification':
                    # NOTE: this `header_txt` must be the same as the one used during the creation
                    #  of the text in run_keras_training()
                    header_txt = "Classification report:"
                    results, classif_report = result_txt.split(header_txt)
                    classif_report = header_txt + classif_report
                    st.info(results)
                    st.text(classif_report)
                else:
                    st.info(result_txt)

        if self.deployment_type == 'Image Classification' and \
                self.training_path['confusion_matrix_file'].exists():
            logger.debug("Showing confusion matrix from "
                         f"{self.training_path['confusion_matrix_file']}")
            image = plt.imread(self.training_path['confusion_matrix_file'])
            st.image(image)
            st.markdown("___")

        # ************* Show predictions on test set images *************
        st.subheader("Prediction Results on Validation/Test Set:")

        # load back the best model
        logger.info(f"Loading trained Keras model for {self.deployment_type}")
        if not hasattr(self, 'model'):
            if self.is_not_pretrained:
                self.model = load_trained_keras_model(
                    self.training_path['output_keras_model_file'])
            else:
                self.model = load_keras_model(self.training_path['output_keras_model_file'],
                                              self.metrics, self.training_param)
            if self.deployment_type == 'Image Classification':
                # use preprocess_fn if available
                self.preprocess_fn: Callable = self.get_preprocess_fn(
                    self.model)

        options_col, _ = st.columns([1, 1])
        prev_btn_col_1, next_btn_col_1, _ = st.columns([1, 1, 3])
        if self.deployment_type == 'Image Classification':
            image_col, image_col_2 = st.columns([1, 1])
        else:
            figure_row_place = st.container()
        prev_btn_col_2, next_btn_col_2, _ = st.columns([1, 1, 3])

        if 'start_idx' not in session_state:
            # to keep track of the image index to show
            session_state['start_idx'] = 0

        logger.info("Loading test set data")
        if self.deployment_type == 'Image Classification':
            X_test, y_test, encoded_label_dict = get_test_images_labels(
                self.training_path['test_set_pkl_file'],
                self.deployment_type
            )
        else:
            X_test, y_test = get_test_images_labels(
                self.training_path['test_set_pkl_file'],
                self.deployment_type
            )

            # the generated masks are required for evaluation
            if not (self.dataset_export_path / 'masks').exists():
                with st.spinner("Exporting labeled data for evaluation ..."):
                    logger.info("Exporting tasks for evaluation ...")
                    session_state.project.export_tasks(
                        for_training_id=self.training_id)

        with options_col:
            st.header("Prediction results on test set")

            def reset_start_idx():
                session_state['start_idx'] = 0

            st.number_input(
                "Number of samples to show:",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                format='%d',
                key='n_samples',
                help="Number of samples to run inference and display results.",
                on_change=reset_start_idx
            )

            total_samples = len(y_test)
            st.info(f"**Total test set images**: {total_samples}")

        n_samples = int(session_state['n_samples'])
        start_idx = session_state['start_idx']
        # st.write(f"{start_idx = }")
        # st.write(f"{start_idx + n_samples = }")
        current_image_paths = X_test[start_idx: start_idx + n_samples]
        # NOTE: these are mask_paths for segmentation task
        current_labels = y_test[start_idx: start_idx + n_samples]

        start = start_idx + 1
        end = start_idx + n_samples
        end = end if end <= total_samples else total_samples
        logger.info(f"Detecting from the test set images: {start}"
                    f" to {end} ...")
        options_col.info(
            f"Showing sample images: **{start}** to **{end}**")

        def previous_samples():
            if session_state['start_idx'] > 0:
                session_state['start_idx'] -= n_samples

        def next_samples():
            max_start = total_samples - n_samples
            if session_state['start_idx'] < max_start:
                session_state['start_idx'] += n_samples

        prev_btn_col_1.button('⏮️ Previous samples', key='btn_prev_images_1',
                              on_click=previous_samples)
        next_btn_col_1.button('Next samples ⏭️', key='btn_next_images_1',
                              on_click=next_samples)

        image_size = self.training_param["image_size"]
        if self.deployment_type == 'Image Classification':
            image_cols = cycle((image_col, image_col_2))
            with st.spinner("Running classifications ..."):
                for i, (p, label, col) in enumerate(zip(
                        current_image_paths, current_labels, image_cols)):
                    logger.debug(f"Image path: {p}")
                    filename = os.path.basename(p)

                    img = cv2.imread(p)
                    start_t = perf_counter()
                    results = classification_inference_pipeline(
                        img, model=self.model, image_size=image_size,
                        encoded_label_dict=encoded_label_dict,
                        preprocess_fn=self.preprocess_fn)
                    pred_classname = results['pred_classname']
                    y_proba = results['y_proba']
                    time_elapsed = perf_counter() - start_t
                    logger.info(f"Inference on image: {filename} "
                                f"[{time_elapsed:.4f}s]")

                    true_classname = encoded_label_dict[label]

                    caption = (f"{start + i}. {filename}; "
                               f"Actual: {true_classname}; "
                               f"Predicted: {pred_classname}; "
                               f"Score: {y_proba * 100:.1f}")

                    with col:
                        st.image(img, channels='BGR', caption=caption)
        else:
            with figure_row_place:
                class_colors = create_class_colors(self.class_names)
                ignore_background = st.checkbox(
                    "Ignore background", value=True, key='ignore_background',
                    help="Ignore background class for visualization purposes")
                legend = create_color_legend(
                    class_colors, bgr2rgb=False, ignore_background=ignore_background)
                st.markdown("**Legend**")
                st.image(legend)
            # convert to array
            class_colors = np.array(list(class_colors.values()),
                                    dtype=np.uint8)
            with st.spinner("Running segmentation ..."):
                for i, (img_path, mask_path) in enumerate(zip(current_image_paths, current_labels)):
                    logger.debug(f"Image path: {img_path}")
                    logger.debug(f"Mask path: {mask_path}")
                    filename = os.path.basename(img_path)

                    image = cv2.imread(img_path)
                    start_t = perf_counter()
                    results = segment_inference_pipeline(
                        image, model=self.model, image_size=image_size,
                        class_colors=class_colors,
                        ignore_background=ignore_background)
                    pred_output = results['img']
                    time_elapsed = perf_counter() - start_t
                    logger.info(f"Inference on image: {filename} "
                                f"[{time_elapsed:.4f}s]")

                    # convert to RGB for visualizing with Matplotlib
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    figure_row_place.subheader(
                        f"Image {start + i}: {filename}")
                    fig = plt.figure()
                    plt.subplot(131)
                    plt.title("Original Image")
                    plt.imshow(image)
                    plt.axis('off')

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    plt.subplot(132)
                    plt.title("Ground Truth")
                    true_output = get_colored_mask_image(
                        image, mask, class_colors,
                        ignore_background=ignore_background)
                    plt.imshow(true_output)
                    plt.axis('off')

                    plt.subplot(133)
                    plt.title("Predicted")
                    plt.imshow(pred_output)
                    plt.axis('off')

                    plt.tight_layout()
                    figure_row_place.pyplot(fig)
                    figure_row_place.markdown("___")

                unique_pixels = np.unique(mask)
                # logger.debug(f"{unique_pixels = }")

        prev_btn_col_2.button('⏮️ Previous samples', key='btn_prev_images_2',
                              on_click=previous_samples)
        next_btn_col_2.button('Next samples ⏭️', key='btn_next_images_2',
                              on_click=next_samples)

    def export_keras_model(self):
        paths = self.training_path

        os.makedirs(paths['export'], exist_ok=True)
        for fpath in (paths['labelmap_file'], paths['output_keras_model_file']):
            try:
                shutil.copy2(fpath, paths['export'])
            except Exception as e:
                logger.error(f"Something wrong when copying the file!: {e}")

        tarfile_path = paths['model_tarfile']
        with st.spinner("Creating model tarfile to download ..."):
            create_tarfile(tarfile_path.name,
                           target_path=paths['export'],
                           dest_dir=tarfile_path)
        logger.debug("Exported tarfile for Keras model at: "
                     f"{tarfile_path}")
