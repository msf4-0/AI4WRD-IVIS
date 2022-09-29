"""
Title: Labelmap Generator
Date: 8/9/2021
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

import sys
from enum import IntEnum
from pathlib import Path
from typing import IO, List, Union

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.file_handler import get_member
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import get_identifier_str_IntEnum, split_string
from core.utils.log import logger
from data_manager.database_manager import init_connection
from deployment.deployment_management import (COMPUTER_VISION_LIST, Deployment,
                                              DeploymentType)
# ************** TensorFLow ***************************
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap, StringIntLabelMapItem)
from object_detection.utils.label_map_util import (
    _validate_label_map, convert_label_map_to_categories,
    create_categories_from_labelmap)
from path_desc import chdir_root

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<


class Framework(IntEnum):
    TensorFlow = 0
    PyTorch = 1
    Scikit_learn = 2
    Caffe = 3
    MXNet = 4
    ONNX = 5
    YOLO = 6

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Framework[s]
        except KeyError:
            raise ValueError()


# Names to for file storing Labelmaps
LABELMAP_NAME = {
    Framework.TensorFlow: 'labelmap.pbtxt',
    Framework.PyTorch: 'labelmap.json',  # pickle,json or txt
    Framework.Scikit_learn: '',
    Framework.Caffe: '',
    Framework.MXNet: '',
    Framework.ONNX: '',
    Framework.YOLO: 'labelmap.txt',

}
FRAMEWORK = {
    "TensorFlow": Framework.TensorFlow,
    "PyTorch": Framework.PyTorch,
    "Scikit-learn": Framework.Scikit_learn,
    "Caffe": Framework.Caffe,
    "MXNet": Framework.MXNet,
    "ONNX": Framework.ONNX,
    "YOLO": Framework.YOLO
}
# <<<< Variable Declaration <<<<


def get_framework(framework: Union[str, Framework], string: bool = False) -> Union[str, Framework]:
    """Get Framework string or IntEnum constants

    Args:
        framework (Union[str, Framework]): Framework string or IntEnum constant
        string (bool, optional): True if to obtain type string, False to obtain IntEnum constant. Defaults to False.

    Returns:
        Union[str, Framework]: Converted Framework
    """

    assert isinstance(
        framework, (str, Framework)), f"framework must be String or IntEnum"

    model_type = get_identifier_str_IntEnum(
        framework, Framework, FRAMEWORK, string=string)

    return model_type


class Labels:

    def __init__(self) -> None:
        self.filename: str = None
        self.filePath: Path = None
        self.dict: List = []
        self.label_map_string: str = None

    @staticmethod
    def generate_list_of_labels(comma_separated_string: str) -> List[str]:
        """Generate a List of labels from comma-separated string

        Args:
            comma_separated_string (str): Comma-separated string

        Returns:
            List: List of labels
        """
        # NOTE: `set` does not preserve the order of the labels
        labels_list = [remove_newline_trailing_whitespace(x)
                       for x in split_string(comma_separated_string, ',')]

        return labels_list

    def generate_labelmap_string(labels_list: List[str],
                                 framework: Union[str, Framework],
                                 deployment_type: Union[str, DeploymentType]
                                 ) -> str:
        """Generate String of labelmap based on Framework and Deep Learning Architectures*
        #### Currently used in Computer Vision Applications
        * Currently only supports TensorFlow

        Args:
            labels_list (List[str]): List of Labels eg. ['car','tarmac','airport']
            framework (Union[str, Framework]): Deep Learning Framework of Model
            deployment_type (Union[str, DeploymentType]): Deployment type of Model

        Returns:
            str: Labelmap string
        """
        # Generate string of labelmap
        # strip comma separated string

        framework = get_framework(framework=framework,
                                  string=False)
        deployment_type = Deployment.get_deployment_type(deployment_type=deployment_type,
                                                         string=False)

        if framework == Framework.TensorFlow:

            if deployment_type in COMPUTER_VISION_LIST:
                labelmap_string = TensorFlow.label_map_to_text(labels_list)

                return labelmap_string

    @staticmethod
    def generate_labelmap_file(labelmap_string: str,
                               dst: Union[str, Path],
                               framework: Union[str, Framework],
                               deployment_type: Union[str, DeploymentType]):
        """Generate labelmap text file from labelmap string

        Args:
            labelmap_string (str): Labelmap string from generate_labelmap_string
            dst (Union[str, Path]): Destination of the labelmap file
            framework (Union[str, Framework]): Deep Learning Framework of the Model
            deployment_type (Union[str, DeploymentType]): Deployment type of the Model

        """

        framework = get_framework(framework=framework,
                                  string=False)
        deployment_type = Deployment.get_deployment_type(deployment_type=deployment_type,
                                                         string=False)
        dst = Path(dst)

        assert dst.is_dir(
        ), f"Destination directory is not found for {str(dst)} "

        if framework == Framework.TensorFlow:

            if deployment_type in COMPUTER_VISION_LIST:
                # NOTE: this labelmap filename is closely related to Training.get_paths()
                filepath = dst / 'labelmap.pbtxt'
                TensorFlow.label_map_to_pbtxt(labelmap_text=labelmap_string,
                                              filepath=filepath)

    @staticmethod
    def get_labelmap_member_from_archived(name: str,
                                          archived_filepath: Path = None,
                                          file_object: IO = None,
                                          decode: str = 'utf-8') -> str:
        """Generate labelmap string if labelmap file is a member of an archived file.
        Contents are extracted without unpacking the contents of the archived file.

        Args:
            name (str): Name of the labelmap file
            archived_filepath (Path, optional): Path to the archived file. Defaults to None.
            file_object (IO, optional): File-like object or io_Buffer. Defaults to None.
            decode (str, optional): Decoding format. Defaults to 'utf-8'.

        Returns:
            str: Labelmap string
        """
        label_map_string = get_member(name=name,
                                      archived_filepath=archived_filepath,
                                      file_object=file_object,
                                      decode=decode)
        return label_map_string

    @staticmethod
    def generate_labelmap_dict(label_map_string: str, framework: Union[str, Framework]) -> List:
        """Generate dictionary of labelmap with label value as key and class id as value

        Args:
            label_map_string (str): Labelmap string
            framework (Union[str, Framework]): Deep Learning Framework of the Model

        Returns:
            List: List of Dictionary of Labels
        """
        label_map_dict = []
        framework = get_framework(framework=framework,
                                  string=False)
        if framework == Framework.TensorFlow:
            label_map_dict = TensorFlow.read_labelmap_file(
                label_map_string=label_map_string)

        return label_map_dict

    @staticmethod
    def set_num_classes(num_classes: int, config_path: Path = None,
                        pipeline_config: pipeline_pb2.TrainEvalPipelineConfig = None
                        ) -> pipeline_pb2.TrainEvalPipelineConfig:
        """Search for the model field in the pipeline config and set the number of classes."""

        assert any((config_path, pipeline_config)), (
            "Must provide either path or pipeline_config")

        if not pipeline_config:
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            import tensorflow as tf
            with tf.io.gfile.GFile(config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

        # search for the model field
        model_fields = ('ssd', 'faster_rcnn', 'center_net')
        for field in model_fields:
            if pipeline_config.model.HasField(field):
                found_model_field = field
                break
        else:
            logger.error(
                "Cannot find the correct model from the pipeline.config file")
            st.error("Cannot find the correct model from the pipeline.config file. "
                     f"The model field should be one of {model_fields}")
            st.stop()
        logger.info(f"The found model field is '{found_model_field}'")

        getattr(pipeline_config.model,
                found_model_field).num_classes = num_classes
        return pipeline_config

    @staticmethod
    def find_optimizer_field(pipeline_config: pipeline_pb2.TrainEvalPipelineConfig):
        available_optimizers = (
            x for x in dir(pipeline_config.train_config.optimizer) 
            if 'optimizer' in x)
        for opt in available_optimizers:
            if pipeline_config.train_config.optimizer.HasField(opt):
                return opt
        raise Exception("Cannot find the any optimizer in the pipeline.config file")


class TensorFlow(object):

    @staticmethod
    def label_map_to_text(classes: List, start=1) -> str:
        """Generate labelmap string

        Args:
            classes (List): List of labels/classes
            start (int, optional): First index excluding Background class of the labelmap. Defaults to 1.

        Returns:
            str: Labelmap string
        """
        # 'id' must start from 1
        msg = StringIntLabelMap()
        for id, name in enumerate(classes, start=start):
            msg.item.append(StringIntLabelMapItem(
                id=id, name=name))

        text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
        return text

    @staticmethod
    def label_map_to_pbtxt(labelmap_text: bytes, filepath: Path) -> None:
        """Generate TensorFlow 2 OD API Labelmap Protocol Buffer Text (.pbtxt)

        Args:
            labelmap_text (bytes): Labelmap string
            filepath (Path): Path to labelmap file to be stored
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            f.write(labelmap_text)

    @staticmethod
    def read_labelmap_file(label_map_path: Union[str, Path] = None,
                           label_map_string: str = None,
                           use_display_name: bool = True) -> List:
        """Create dictionary of labels/classes 

        Args:
            label_map_path (Union[str, Path], optional): Path to labelmap.pbtxt. Defaults to None.
            label_map_string (str, optional): Labelmap string if label_map_path not set. Defaults to None.
            use_display_name (bool, optional): Use display name stated in the labelmap file. Defaults to True.

        Returns:
            List: List of Labels
        """
        if label_map_path:
            label_map_dict = create_categories_from_labelmap(
                label_map_path=str(label_map_path))

        elif label_map_string:
            label_map = TensorFlow.load_labelmap(
                label_map_string=label_map_string)
            max_num_classes = max(item.id for item in label_map.item)
            label_map_dict = convert_label_map_to_categories(label_map=label_map,
                                                             max_num_classes=max_num_classes,
                                                             use_display_name=use_display_name)

        else:
            label_map_dict = None

        return label_map_dict

    @staticmethod
    def load_labelmap(label_map_string: str) -> List:
        """Parse Labelmap into Protobuf object

        Args:
            label_map_string (str): Labelmap string

        Returns:
            str: String of labelmap dictionaries
        """
        label_map = StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)

        return label_map


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
