"""
Title: Deployment Management
Date: 28/7/2021
Author: Chu Zhen Hao
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
import gc
from json.decoder import JSONDecodeError
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, List, Dict, TYPE_CHECKING
from enum import IntEnum
import json

from psycopg2 import sql
import cv2
import numpy as np
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state
import tensorflow as tf


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from core.utils.helper import Timer, get_identifier_str_IntEnum, get_now_string
from data_manager.database_manager import init_connection, db_fetchall
from core.utils.form_manager import reset_page_attributes
from machine_learning.utils import (
    classification_predict, find_architecture_name, get_classif_model_preprocess_func, get_label_dict_from_labelmap,
    load_keras_model, load_labelmap, load_tfod_model, load_trained_keras_model,
    preprocess_image, segmentation_predict, tfod_detect)
if TYPE_CHECKING:
    from machine_learning.trainer import Trainer
    from training.model_management import Model
from machine_learning.visuals import create_class_colors, draw_tfod_bboxes, get_colored_mask_image
from machine_learning.command_utils import export_tfod_savedmodel
from deployment.utils import (
    classification_inference_pipeline, reset_video_deployment, reset_client,
    reset_csv_file_and_writer, reset_record_and_vid_writer, segment_inference_pipeline, tfod_inference_pipeline)
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>

# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


class DeploymentType(IntEnum):
    Image_Classification = 1
    OD = 2
    Instance = 3
    Semantic = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentType[s]
        except KeyError:
            raise ValueError()


class DeploymentPagination(IntEnum):
    Models = 0
    UploadModel = 1
    Deployment = 2
    SwitchUser = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DeploymentPagination[s]
        except KeyError:
            raise ValueError()


@dataclass(eq=False)
class DeploymentConfig:
    timezone: str = "Singapore"
    input_type: str = 'Image'  # or Video
    video_type: str = 'Uploaded Video'  # Or Video Camera or From MQTT
    use_multi_cam: bool = False
    num_cameras: int = 1
    video_width: int = 640
    # use_camera: bool = False
    camera_types: List[str] = field(default_factory=list)  # USB or IP Camera
    camera_sources: List[Union[int, str]] = field(default_factory=list)
    # to store the session_state key names
    camera_keys: List[str] = field(default_factory=list)
    # camera_ports: List[int] = field(default_factory=lambda: [0, 1])
    # to store the title/view for multiple cameras
    # camera_titles: Dict[int, str] = field(default_factory=dict)
    camera_titles: List[str] = field(default_factory=lambda: [''])
    retention_period: int = 7
    # whether is publishing inference results or not
    publishing: bool = True
    # whether is publishing current output frame or not
    publish_frame: bool = False

    # ip_cam_addresses: List[str] = field(default_factory=list)
    # for TFOD
    confidence_threshold: float = 0.7
    # for segmentation
    ignore_background: bool = False

    def asdict(self):
        return deepcopy(self.__dict__)


# KIV
DEPLOYMENT_TYPE = {
    "Image Classification": DeploymentType.Image_Classification,
    "Object Detection with Bounding Boxes": DeploymentType.OD,
    "Semantic Segmentation with Polygons": DeploymentType.Semantic,
    "Semantic Segmentation with Masks": DeploymentType.Instance
}

COMPUTER_VISION_LIST = [DeploymentType.Image_Classification, DeploymentType.OD,
                        DeploymentType.Instance, DeploymentType.Semantic]
# <<<< Variable Declaration <<<<

CSV_DATETIME_FMT = "%d-%b-%Y"


class BaseDeployment:
    def __init__(self) -> None:
        self.id: int = None
        self.name: str = None
        self.model_selected = None


class Deployment(BaseDeployment):

    def __init__(self,
                 project_path: Path,
                 deployment_type: str,
                 training_path: Dict[str, Path],
                 image_size: int = None,
                 metrics: List[Callable] = None,
                 metric_names: List[str] = None,
                 class_names: List[str] = None,
                 is_uploaded: bool = False,
                 category_index: Dict[int, Dict[str, Any]] = None,
                 training_param: Dict[str, Any] = None,
                 preprocess_fn: Callable = None,
                 architecture_name: str = None) -> None:
        super().__init__()
        # project_path is currently only used to get the path to save CSV file
        # or captured frames
        self.project_path = project_path
        self.csv_save_dir: Path = self.project_path / 'deployment_results'
        self.deployment_type = deployment_type
        self.training_path = training_path

        # these are needed for classification/segmentation model
        self.image_size = image_size
        self.metrics = metrics
        self.metric_names = metric_names
        self.class_names = class_names

        # preprocess_input function for classification, should be obtained
        # from self.run_preparation_pipeline()
        self.preprocess_fn = preprocess_fn
        # need attached model name to get the preprocess function
        # will skip preprocessing if not provided
        self.architecture_name = architecture_name

        # for segmentation to get results faster
        self.class_names_arr: np.ndarray = np.array(class_names)
        # only needed for segmentation to load the model
        self.training_param = training_param

        # whether is user-uploaded model
        self.is_uploaded = is_uploaded

        # only needed for TFOD
        self.category_index = category_index

        # not needed for now
        # self.deployment_list: List = self.query_deployment_list()

        # to store the loaded model later
        self.model: tf.keras.Model = None

    @classmethod
    def from_trainer(cls, trainer: Trainer):
        project_path = trainer.project_path
        deployment_type = trainer.deployment_type
        training_path = trainer.training_path
        metrics = trainer.metrics
        metric_names = trainer.metric_names
        # maybe class_names can get from labelmap directly for user-uploaded models??
        class_names = trainer.class_names
        # need this name to get preprocess_input function for classification model
        architecture_name = trainer.attached_model_name
        if deployment_type != 'Object Detection with Bounding Boxes':
            image_size = trainer.training_param['image_size']
        else:
            image_size = None
        if deployment_type == 'Semantic Segmentation with Polygons':
            training_param = trainer.training_param
        else:
            training_param = None
        return cls(project_path, deployment_type, training_path, image_size,
                   metrics, metric_names, class_names,
                   architecture_name=architecture_name,
                   training_param=training_param)

    @classmethod
    def from_uploaded_model(cls, model: Model, uploaded_model_dir: Path,
                            category_index: Dict[int, Dict[str, Any]]):
        project_path = session_state.project.get_project_path(
            session_state.project.name)
        deployment_type = model.deployment_type
        training_path = {'uploaded_model_dir': uploaded_model_dir}
        class_names = [d['name'] for d in category_index.values()]
        if deployment_type != 'Object Detection with Bounding Boxes':
            # don't need category_index for classification/segmentation
            category_index = None
        return cls(project_path, deployment_type, training_path,
                   category_index=category_index, class_names=class_names,
                   is_uploaded=True)

    @staticmethod
    @st.experimental_memo
    def query_deployment_list():
        query_deployment_list_sql = """
                                    SELECT
                                        name
                                    FROM
                                        deployment_type
                                    ORDER BY
                                        id ASC;
                                    """
        deployment_list = db_fetchall(query_deployment_list_sql, conn)
        return deployment_list if deployment_list else None

    def query_model_table(self, model_table) -> NamedTuple:
        schema, table = [x for x in model_table.split('.')]
        query_model_table_SQL = sql.SQL("""SELECT
                m.id AS "ID",
                m.name AS "Name",
                f.name AS "Framework",
                m.model_path AS "Model Path"
            FROM
                {table} m
                LEFT JOIN public.framework f ON f.id = m.framework_id
                where m.deployment_id = (SELECT id from public.deployment_type where name = %s);""").format(table=sql.Identifier(schema, table))
        query_model_table_vars = [self.name]
        return_all = db_fetchall(
            query_model_table_SQL, conn, query_model_table_vars, fetch_col_name=True)
        if return_all:
            project_model_list, column_names = return_all
        else:
            project_model_list = []
            column_names = []
        return project_model_list, column_names

    @staticmethod
    def get_deployment_type(deployment_type: Union[str, DeploymentType], string: bool = False):

        assert isinstance(
            deployment_type, (str, DeploymentType)), f"deployment_type must be String or IntEnum"

        deployment_type = get_identifier_str_IntEnum(
            deployment_type, DeploymentType, DEPLOYMENT_TYPE, string=string)

        return deployment_type

    def run_preparation_pipeline(self, re_export: bool = False):
        paths = self.training_path
        if self.deployment_type == 'Object Detection with Bounding Boxes':
            if self.is_uploaded:
                # 'uploaded_model_dir' is generated in `self.from_uploaded_model()`;
                # category_index has already been loaded in `self.from_uploaded_model()`
                saved_model_dir = next(
                    paths['uploaded_model_dir'].rglob("saved_model"))
            else:
                # this is a project model trained in our app
                export_tfod_savedmodel(paths, re_export=re_export)
                saved_model_dir = paths['export'] / 'saved_model'
                self.category_index = load_labelmap(
                    paths['labelmap_file'])

            self.model = load_tfod_model(saved_model_dir)
        else:
            if self.is_uploaded:
                model_fpath = next(paths['uploaded_model_dir'].rglob('*.h5'))
                self.model = load_trained_keras_model(model_fpath)
                # attempt to find the uploaded model's architecture name
                # to use for preprocess_input function
                self.architecture_name = find_architecture_name(self.model)
                self.preprocess_fn = get_classif_model_preprocess_func(
                    self.architecture_name)
            else:
                model_fpath = paths['output_keras_model_file']
                self.model = load_keras_model(model_fpath, self.metrics,
                                              self.training_param)
                # delete it after loaded the model
                delattr(self, 'training_param')

            # take the width as the image_size for preprocessing
            # as we resize the image with the same width and height
            self.image_size = self.model.input_shape[1]

        if self.deployment_type == 'Image Classification':
            if self.is_uploaded:
                # 'uploaded_model_dir' is generated in `self.from_uploaded_model()`
                labelmap_path = next(
                    paths['uploaded_model_dir'].rglob("*.pbtxt"))
            else:
                labelmap_path = paths['labelmap_file']
            self.encoded_label_dict = get_label_dict_from_labelmap(
                labelmap_path)
        elif self.deployment_type == 'Semantic Segmentation with Polygons':
            self.class_colors = create_class_colors(self.class_names)
            # convert to array to use for segment_inference_pipeline()
            self.class_colors_arr: np.ndarray = np.array(
                list(self.class_colors.values()), dtype=np.uint8)

        tf.keras.backend.clear_session()
        gc.collect()

    def get_inference_pipeline(self, **kwargs) -> Callable[..., Dict[str, Any]]:
        assert 'img' not in kwargs, "Image should only be passed in during inference time"
        if self.deployment_type == 'Image Classification':
            return partial(
                classification_inference_pipeline, model=self.model,
                image_size=self.image_size, preprocess_fn=self.preprocess_fn,
                encoded_label_dict=self.encoded_label_dict, **kwargs)
        elif self.deployment_type == 'Object Detection with Bounding Boxes':
            return partial(
                tfod_inference_pipeline, model=self.model,
                category_index=self.category_index, is_checkpoint=False, **kwargs)
        elif self.deployment_type == 'Semantic Segmentation with Polygons':
            return partial(
                segment_inference_pipeline, model=self.model,
                image_size=self.image_size, class_colors=self.class_colors_arr,
                **kwargs)

    def get_classification_results(self, pred_classname: str, probability: float,
                                   timezone: str, camera_title: str = '', **kwargs):
        results = [{'name': pred_classname,
                    # need to change to string to be serialized with json.dumps()
                    'probability': f"{probability * 100:.2f}%",
                    'view': camera_title,
                    'time': get_now_string(timezone=timezone)}]
        return results

    def get_detection_results(self, detections: Dict[str, Any], timezone: str,
                              camera_title: str = '',
                              get_bbox_coords: bool = False,
                              conf_threshold: float = None, **kwargs) -> List[Dict[str, Any]]:
        results = []
        now = get_now_string(timezone=timezone)
        for class_id, prob, box in zip(detections['detection_classes'],
                                       detections['detection_scores'],
                                       detections['detection_boxes']):
            if prob < conf_threshold:
                continue
            if get_bbox_coords:
                ymin, xmin, ymax, xmax = np.around(box, 4).astype(str).tolist()
                tl = [xmin, ymin]
                br = [xmax, ymax]
                detection = {'name': self.category_index[class_id]['name'],
                             # need to change to string to be serialized with json.dumps()
                             'probability': f"{prob * 100:.2f}%",
                             'top_left': tl,
                             'bottom_right': br,
                             'view': camera_title,
                             'time': now}
            else:
                detection = {'name': self.category_index[class_id]['name'],
                             'probability': f"{prob * 100:.2f}%",
                             'view': camera_title,
                             'time': now}
            results.append(detection)
        return results

    def get_segmentation_results(
            self, prediction_mask: np.ndarray,
            timezone: str, camera_title: str = '', **kwargs) -> List[Dict[str, Any]]:
        class_names = self.class_names_arr[np.unique(prediction_mask)]
        results = [{'classes_found': class_names.tolist(),
                    'view': camera_title,
                   'time': get_now_string(timezone=timezone)}]
        return results

    def get_csv_path(self, now: datetime) -> Path:
        full_date = now.strftime(CSV_DATETIME_FMT)
        csv_path = self.csv_save_dir / full_date / f"{full_date}.csv"
        return csv_path

    def get_frame_save_dir(self, save_type: str = 'video') -> Path:
        """To save frames or record videos. 

        `save_type` should be either `'video'`, 'NG', or `'image'.`"""
        if save_type == 'video':
            dirname = 'video-recordings'
        elif save_type == 'NG':
            dirname = 'NG-images'
        elif save_type == 'csv-labels':
            dirname = 'labels_to_check'
        else:
            dirname = 'saved-frames'
        record_dir = self.project_path / dirname
        return record_dir

    @staticmethod
    def get_datetime_from_csv_path(csv_path: Path) -> datetime:
        """`csv_path` can be the folder name or the filename"""
        full_date = csv_path.stem
        dt_format = CSV_DATETIME_FMT
        dt = datetime.strptime(full_date, dt_format)
        return dt

    def delete_old_csv_files(self, retention_period: int):
        """Delete CSV files older than `retention_period` (in `days`)."""
        csv_paths = self.csv_save_dir.iterdir()
        sorted_csv_paths = sorted(
            csv_paths, key=self.get_datetime_from_csv_path, reverse=True)
        now = datetime.now()
        for p in sorted_csv_paths:
            csv_date = self.get_datetime_from_csv_path(p)
            days_from_created = (now - csv_date).days
            if days_from_created > retention_period:
                logger.info(f"Removing old CSV file older than {retention_period} "
                            f"days at {p}")
                shutil.rmtree(p)
            else:
                # stop because no more older files
                break

    @staticmethod
    def get_camera_views_from_titles(cam_titles: List[str]) -> Dict[str, int]:
        """One camera title can have multiple views separated by a frontslash '/'.

        Returns:
            Dict[str, int]: Camera view -> camera index
        """
        camera_view2idx = {}
        for i, view in enumerate(cam_titles):
            if '/' in view:
                # this is for multiple views for a single camera
                all_views = view.split('/')
                for v in all_views:
                    v = v.strip().lower()
                    camera_view2idx[v] = i
            else:
                view = view.strip().lower()
                camera_view2idx[view] = i
        return camera_view2idx

    @staticmethod
    def validate_received_label_msg(msg_payload: bytes) -> Union[bool, Tuple[str, List[str]]]:
        """
        Validate the received message payload for label checking to have the correct 
        JSON format.

        e.g. {"labels": ["label1", "label2"], "view": "top"}

        Returns:
            - If format is not valid, returns False.
            - If format is valid, returns a Tuple[str, List[str]] for the 
                (view, required_labels).
        """
        try:
            mqtt_recv = json.loads(msg_payload)
        except JSONDecodeError as e:
            logger.error(f'Error reading JSON object "{msg_payload}" from the received '
                         'MQTT message payload. Please pass in the correct JSON format as '
                         f'specified, e.g. {{"labels": ["label1", "label2"], "view": "top"}}. '
                         f'Error: {e}')
            return False

        required_labels = mqtt_recv.get('labels')
        view = mqtt_recv.get('view')

        if not view:
            logger.error(
                '"view" key is not found from the received JSON object')
            return False
        if view == 'end':
            # no need to check required labels
            return view, required_labels

        if not required_labels:
            logger.error(
                '"labels" key is not found from the received JSON object')
            return False
        if not isinstance(view, str):
            logger.error(f'The received value for the "view" key '
                         f'"{view}" is not in string format')
            return False
        if not isinstance(required_labels, list):
            logger.error(f'The received value for the "labels" key '
                         f'"{required_labels}" is not in List format')
            return False
        return view, required_labels

    @staticmethod
    def check_labels(results: List[Dict[str, Any]],
                     required_labels: List[str],
                     deployment_type: str) -> bool:
        """Check the `results` to see whether all `required_labels` 
        are found in the `results`, which are generated from the get_*_results()
        function for each deployment type.
        """
        if deployment_type == 'Semantic Segmentation with Polygons':
            detected_labels: List[str] = results[0]['classes_found'].copy()
        else:
            detected_labels: List[str] = [r['name'] for r in results]

        logger.info(f"Required labels: {required_labels}")
        logger.info(f"Detected labels: {detected_labels}")

        for label in required_labels:
            if label not in detected_labels:
                return False
            # remove it from the list to be able to count the exact number of labels
            detected_labels.remove(label)
        return True

    @staticmethod
    def reset_deployment_page():
        """Method to reset all widgets and attributes in the Deployment Pages when changing pages
        """
        tf.keras.backend.clear_session()

        reset_video_deployment()
        reset_record_and_vid_writer()
        reset_csv_file_and_writer()
        reset_client()

        project_attributes = [
            "deployment_pagination", "deployment", "trainer", "publishing",
            "refresh", "deployment_conf", "today", "mqtt_conf",
            "image_idx", "check_labels", "working_ports"
        ]

        reset_page_attributes(project_attributes)

        # run garbage collect at the end
        gc.collect()


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
