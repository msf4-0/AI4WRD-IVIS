"""
Title: Dataset Management
Date: 18/7/2021
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

import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from base64 import b64encode
from datetime import datetime
from enum import IntEnum
from io import BytesIO
from mimetypes import guess_type
from operator import itemgetter
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from core.utils.code_generator import get_random_string
from core.utils.dataset_handler import get_image_size
from core.utils.file_handler import (
    IMAGE_EXTENSIONS,
    create_folder_if_not_exist,
    extract_one_to_bytes,
)
from core.utils.form_manager import (
    check_if_exists,
    check_if_field_empty,
    reset_page_attributes,
)
from core.utils.helper import (
    get_directory_name,
    get_filetype,
    get_mime,
    reset_camera_and_ports,
)
from core.utils.log import logger
from data_manager.database_manager import db_fetchall, db_fetchone, init_connection
from imutils.paths import list_images
from natsort import os_sorted

# >>>> User-defined Modules >>>>
from path_desc import BASE_DATA_DIR, CAPTURED_IMAGES_DIR, DATASET_DIR, TEMP_DIR
from PIL import Image
from stqdm import stqdm
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit.uploaded_file_manager import UploadedFile
from videoprops import get_audio_properties, get_video_properties

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# >>>> Variable Declaration <<<<


class DataPermission(IntEnum):
    ViewOnly = 0
    Edit = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DataPermission[s]
        except KeyError:
            raise ValueError()


class DatasetPagination(IntEnum):
    Dashboard = 0
    New = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DatasetPagination[s]
        except KeyError:
            raise ValueError()


class FileTypes(IntEnum):
    Image = 0
    Video = 1
    Audio = 2
    Text = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return FileTypes[s]
        except KeyError:
            raise ValueError()

# <<<< Variable Declaration <<<<


CSV_IMAGE_COL = 'image'
CSV_LABEL_COL = 'label'


def check_required_csv_columns(df: pd.DataFrame):
    required_columns = [CSV_IMAGE_COL, CSV_LABEL_COL]
    ok = set(required_columns).issubset(df.columns)
    if not ok:
        error_txt = "Missing 'image' column or 'label' column (or both) in the CSV file."
        st.error(error_txt)
        logger.error(error_txt)
        st.stop()


def check_invalid_paths(df: pd.DataFrame):
    """Currently used to check for backslash and frontslash only."""
    # the pattern for backslash/frontslash
    backslash_pattern = r"/|\\"
    # first row for filename, second row for label name
    backslash_rows = df[(df.iloc[:, 0].str.contains(
        backslash_pattern)) | (df.iloc[:, 1].str.contains(backslash_pattern))]
    if not backslash_rows.empty:
        # backslash/frontslash is unwanted because we will create the class
        # directories for our training later, and backslash would cause the
        # directories to be incorrect
        st.error(
            "Backslash/frontslash is found in the following image names or "
            "class labels, please remove them as only filename/labelname is required.")
        st.dataframe(backslash_rows)
        st.stop()


def get_items_from_indices(indices: List[int], input_items: Iterable) -> List[Any]:
    if len(indices) == 1:
        # rare case for only one item
        items = [input_items[indices[0]]]
    else:
        items = list(itemgetter(*indices)(input_items))
    return items


def create_classification_result(label: str):
    result = [
        {
            "id": get_random_string(length=6),
            "type": "choices",
            "value": {"choices": [label]},
            "to_name": "image",
            "from_name": "label"
        }
    ]
    return result


def convert_to_ls(x, y, width, height, original_width, original_height):
    return x / original_width * 100.0, y / original_height * 100.0, \
        width / original_width * 100.0, height / original_height * 100


def convert_xml2_ls(xml_filepath: Union[str, Path]) -> Tuple[str, List[Dict[str, Any]]]:
    """Converting from XML file (decoded from Streamlit's UploadedFile)
    to Label Studio JSON result format to be stored in Database.
    Note that Label Studio generates a List of Dict of results."""
    # converting for one XML file, i.e. one image
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    # root = ET.fromstring(xml_str)
    filename = root.find("filename").text
    ori_width = int(root.find("size").find("width").text)
    ori_height = int(root.find("size").find("height").text)
    result = []
    for member in root.findall("object"):
        bndbox = member.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        pixel_width = xmax - xmin
        pixel_height = ymax - ymin
        x, y, width, height = convert_to_ls(xmin, ymin,
                                            pixel_width, pixel_height,
                                            ori_width, ori_height)

        cur_result = {
            "original_width": ori_width,
            "original_height": ori_height,
            "image_rotation": 0,
            "value": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rotation": 0,
                "rectanglelabels": [member.find("name").text]
            },
            "id": get_random_string(length=6),
            "from_name": "label",
            "to_name": "img",
            "type": "rectanglelabels"
        }
        result.append(cur_result)
    return filename, result


def check_coco_json_keys(coco_json: Dict[str, Any]):
    main_key2sub_keys = {
        "images": ["width", "height", "id", "file_name"],
        "categories": ["id", "name"],
        "annotations": ["id", "image_id", "category_id",
                        "segmentation", "bbox", "iscrowd", "area"]}

    for main_key, sub_keys in main_key2sub_keys.items():
        if main_key not in coco_json:
            st.error(f'Not a valid COCO JSON file for segmentation: '
                     f'"{main_key}" not found in COCO JSON!')
            st.stop()
        for sub_key in sub_keys:
            # check only the first Dict within the main_key
            if sub_key not in coco_json[main_key][0]:
                st.error(f'Not a valid COCO JSON file for segmentation: "{sub_key}" '
                         f'not found in COCO JSON\'s "{main_key}" values!')
                st.stop()


def convert_coco2ls_points(segmentation: List[List[float]],
                           original_width: int,
                           original_height: int) -> List[Tuple[float, float]]:
    """Convert the COCO JSON's segmentation values to Label Studio scaled (x, y) values."""
    def scale(x, y):
        x = x * 100.0 / original_width
        y = y * 100.0 / original_height
        return x, y

    # concatenate into a list, with each of 2 consecutive values as tuple(x, y)
    result = list(map(scale,
                      segmentation[0][:-1:2],
                      segmentation[0][1::2]))
    return result


class BaseDataset:
    def __init__(self, id) -> None:
        self.id = id
        self.name: str = ''
        self.desc: str = ''
        self.dataset_size: int = None  # Number of files
        self.dataset_path: Path = None
        self.deployment_id: Union[str, int] = None
        self.filetype: str = "Image"
        self.deployment_type: str = None
        self.dataset: List[str] = []  # to hold new image names from upload
        self.dataset_list = []  # List of existing dataset
        self.dataset_total_filesize = 0  # in byte-size
        self.has_submitted = False
        self.raw_data_dict: Dict = {}  # stores raw data of data

# NOTE DEPRECATED *************************
    def query_deployment_id(self) -> int:
        query_id_SQL = """
                        SELECT
                            id
                        FROM
                            public.deployment_type
                        WHERE
                            name = %s;
                        """
        if self.deployment_type is not None and self.deployment_type != '':

            self.deployment_id = db_fetchone(
                query_id_SQL, conn, [self.deployment_type])[0]
        else:
            self.deployment_id = None

    def check_if_field_empty(self, context: Dict, field_placeholder, name_key: str):
        check_if_exists = self.check_if_exists
        empty_fields = check_if_field_empty(
            context, field_placeholder, name_key, check_if_exists)
        return empty_fields

    def check_if_exists(self, context: Dict, conn) -> bool:
        table = 'public.dataset'

        exists_flag = check_if_exists(
            table, context['column_name'], context['value'], conn)

        return exists_flag

    def dataset_PNG_encoding(
            self, archive_dir: Path, verbose: bool = False,
            all_img_names: Set[str] = None) -> List[str]:
        """Save all the images found in the `archive_dir` into `self.dataset_path`.

        `all_img_names` (Optional[Set[str]]): Provide this to check for existing image names
        to avoid duplicate names. Even if not provided, will still generate random code to
        prepend to the image names and compare with existing generated image names.

        Returns `error_img_paths` (List[str]) for any images that were not saved successfully.
        """
        # archive_dir is the directory that contains the extracted tarfile contents
        # `os_sorted` is used to sort the files just like in file browser
        # to make the order of the files make more sense to the user
        # especially in the labelling pages ('data_labelling.py' & 'labelling_dashboard.py')
        image_paths = os_sorted(list_images(archive_dir))
        if not all_img_names:
            # use Set for faster membership checking
            all_img_names = set()
        if not isinstance(all_img_names, set):
            # just in case... to avoid errors
            all_img_names = set(all_img_names)
        error_img_paths = []
        ori_img_names = []
        new_img_names = []

        for img_path in stqdm(image_paths, unit=self.filetype, ascii='123456789#', st_container=st.sidebar, desc="Uploading images"):
            success, ori_img_name, new_img_name = save_single_image(
                img_path, self.dataset_path,
                all_img_names=all_img_names, verbose=verbose)
            ori_img_names.append(ori_img_name)
            new_img_names.append(new_img_name)
            if not success:
                error_img_paths.append(img_path)
        return ori_img_names, new_img_names, error_img_paths

    def calc_total_filesize(self):
        if self.dataset:
            self.dataset_total_filesize = 0
            for data in self.dataset:
                self.dataset_total_filesize += data.size
            # # To get size in MB
            # self.dataset_total_filesize = bytes_divisor(
            #     self.dataset_total_filesize, -2)
        else:
            self.dataset_total_filesize = 0
        return self.dataset_total_filesize

    @staticmethod
    def get_dataset_path(dataset_name: str) -> Path:
        directory_name = get_directory_name(
            dataset_name)  # change name to lowercase
        # join directory name with '-' dash
        dataset_path = DATASET_DIR / directory_name
        # logger.debug(f"Dataset Path: {dataset_path}")
        return dataset_path

    @staticmethod
    def get_image_paths(
            dataset_name: str, return_names: bool = False) -> List[str]:
        """Get all the image paths from the dataset path.

        If `return_names` is True, return only the image filenames.
        """
        dataset_path = BaseDataset.get_dataset_path(dataset_name)
        image_paths = list_images(dataset_path)
        if return_names:
            return [os.path.basename(p) for p in image_paths]
        return list(image_paths)

    def save_dataset(
            self, archive_dir: Path,
            save_images_to_disk: bool = True) -> Tuple[List[str], List[str], List[str]]:
        """`archive_dir` is the directory that contains the extracted tarfile contents

        If `save_images_to_disk` is True, also returns Tuple of List[str] of
        `ori_img_names`, `new_img_names` (new generated names saved in disk),
        and `error_img_paths` for any images that were not saved successfully.
        """

        # Get absolute dataset folder path
        self.dataset_path = self.get_dataset_path(self.name)

        create_folder_if_not_exist(self.dataset_path)
        if save_images_to_disk:
            existing_image_names = set(
                self.get_image_paths(self.name, return_names=True))
            ori_img_names, new_img_names, error_img_paths = self.dataset_PNG_encoding(
                archive_dir, all_img_names=existing_image_names)
            return ori_img_names, new_img_names, error_img_paths
        return [], [], []
        # st.success(f"Successfully created **{self.name}** dataset")

    @staticmethod
    def delete_dataset(id: int):
        """Delete the dataset from database. This will also delete all the tasks and 
        annotations associated with the dataset (of all associated projects), and remove
        from project_dataset table. Then finally, delete the dataset directory from the 
        system.
        """
        sql_delete = """
                    DELETE 
                    FROM public.dataset 
                    WHERE id = %s
                    RETURNING name;
        """
        delete_vars = [id]
        record = db_fetchone(sql_delete, conn, delete_vars)
        if not record:
            logger.error(f"Error occurred when deleting dataset, "
                         f"cannot find dataset ID: {id}")
            return
        else:
            dataset_name = record.name
        logger.info(f"Deleted existing dataset of ID {id} "
                    f"of name: {dataset_name}")

        dataset_path = Dataset.get_dataset_path(dataset_name)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            logger.info(f"Deleted dataset directory at: {dataset_path}")

    def update_dataset_size(self) -> bool:
        self.dataset_path = self.get_dataset_path(self.name)
        # new_dataset_size = len([file for file in Path(
        #     self.dataset_path).iterdir() if file.is_file()])
        new_dataset_size = len(list(list_images(self.dataset_path)))

        update_dataset_size_SQL = """
                                    UPDATE
                                        public.dataset
                                    SET
                                        dataset_size = %s
                                    WHERE
                                        id = %s
                                    RETURNING dataset_size;
                                    """
        update_dataset_size_vars = [new_dataset_size, self.id]
        new_dataset_size_return = db_fetchone(
            update_dataset_size_SQL, conn, update_dataset_size_vars)
        self.dataset_size = (
            int(new_dataset_size_return.dataset_size)
            if new_dataset_size_return
            else self.dataset_size)
        if new_dataset_size_return:
            logger.info(
                f"Dataset size updated successfully for ID {self.id}. "
                f"New size: {new_dataset_size}")
            return True
        logger.error(f"Dataset size update failed for ID {self.id}")
        return False


class NewDataset(BaseDataset):
    def __init__(self, id) -> None:
        # init BaseDataset -> Temporary dataset ID from random gen
        super().__init__(id)
        self.dataset_total_filesize: int = 0  # in byte-size
        self.archive_dir: Path = TEMP_DIR
        # these will be created in `self.validate_labeled_data` if user choose to upload labeled dataset
        self.image_files: List[str] = None
        self.annotation_files: List[str] = None

    def validate_labeled_data(self,
                              uploaded_archive: UploadedFile,
                              filepaths: List[str],
                              deployment_type: str,
                              classif_annot_type: str = 'CSV file',
                              return_annotations: bool = False
                              ) -> Union[List[str],
                                         Tuple[List[str], List[str]]]:
        """Validate the uploaded archive contents `filepaths`, including both images
        and annotations, then store the relative annotation filepaths (relative to
        root of tarfile) in `self.annotation_files`.

        Object detection should have one XML file for each uploaded image.

        Image classification should have only images with only one CSV file. 
        The first row of CSV file should be the filename with extension, while 
        the second row should be the class label name.

        Image segmentation should have only images with only one COCO JSON file.

        Returns the list of uploaded image files; and if `return_annotations` is True, also returns the
        list of uploaded annotation files (or a list of single file depending on the deployment type).
        """
        error_filepaths = []
        if deployment_type == 'Object Detection with Bounding Boxes':
            xml_idxs = []
            xml_names = []
            img_names_xml = []
            for i, f in enumerate(filepaths):
                if os.path.splitext(f)[-1] in IMAGE_EXTENSIONS:
                    # get the image filename replaced with .xml to
                    # verify that each image has exactly one xml file
                    replaced_name = os.path.splitext(
                        os.path.basename(f))[0] + ".xml"
                    img_names_xml.append(replaced_name)
                elif f.endswith('.xml'):
                    xml_idxs.append(i)
                    xml_names.append(os.path.basename(f))
                else:
                    error_filepaths.append(f)
            if error_filepaths:
                error_filepaths.sort()
                st.error(f"{len(error_filepaths)} unwanted filepaths found: ")
                with st.expander("Unwanted filepaths:"):
                    st.warning("  \n".join(error_filepaths))
                st.stop()
            if not xml_idxs or not img_names_xml:
                if not xml_idxs:
                    st.warning("No XML file uploaded")
                    logger.info("No XML file uploaded ")
                if not img_names_xml:
                    st.warning("No image uploaded")
                    logger.info("No image uploaded ")
                st.stop()
            if len(img_names_xml) != len(xml_idxs):
                st.error(f"""Every image should have its own XML annotation file.
                But found {len(img_names_xml)} image(s) with {len(xml_idxs)} XML file(s).""")

            unknown_xml_files = set(xml_names).difference(img_names_xml)
            unknown_img_files = set(img_names_xml).difference(xml_names)
            if len(unknown_xml_files) != 0 or len(unknown_img_files) != 0:
                if len(unknown_xml_files) != 0:
                    unknown_xml_files = sorted(unknown_xml_files)
                    st.error(f"""Every image should have its own XML annotation file.
                    But found {len(unknown_xml_files)} XML file(s) without XML associated images.""")
                    with st.expander("List of the XML files:"):
                        st.warning("  \n".join(unknown_xml_files))
                if len(unknown_img_files) != 0:
                    unknown_img_files = sorted(unknown_img_files)
                    fnames = [os.path.splitext(f)[0]
                              for f in unknown_img_files]
                    st.error(f"""Every image should have its own XML annotation file.
                    But found {len(unknown_img_files)} image file(s) without associated XML filepaths.""")
                    with st.expander("List of the images (no file extension):"):
                        st.warning("  \n".join(fnames))
                st.stop()

            xml_filepaths = get_items_from_indices(xml_idxs, filepaths)
            image_idxs = set(range(len(filepaths))).difference(xml_idxs)
            img_filepaths = get_items_from_indices(list(image_idxs), filepaths)

            self.annotation_files = xml_filepaths

            if return_annotations:
                return self.annotation_files, img_filepaths
            return img_filepaths

        # ***************** Checking Image Classification and Segmentation data *****************
        elif deployment_type == 'Image Classification':
            if classif_annot_type == 'CSV file':
                # this is actually CSV format
                required_filetype = '.csv'
                filetype_name = 'CSV'
        elif deployment_type == 'Semantic Segmentation with Polygons':
            required_filetype = '.json'
            filetype_name = 'JSON'

        req_file_idx = []
        img_names = []
        img_filepaths = []
        for i, f in enumerate(filepaths):
            if os.path.splitext(f)[-1] in IMAGE_EXTENSIONS:
                img_filepaths.append(f)
                img_names.append(os.path.basename(f))
                if classif_annot_type != 'CSV file':
                    # try extracting label from folder name
                    folder_path = os.path.dirname(f)
                    if not folder_path:
                        txt = (f"'{f}' does not appear to be in a folder, cannot "
                               "extract label from folder name")
                        logger.error(txt)
                        st.error(txt)
                        st.stop()
            elif classif_annot_type == 'CSV file' and f.endswith(required_filetype):
                st.success(f"Found a {filetype_name} file: {f}")
                req_file_idx.append(i)
            else:
                error_filepaths.append(f)
        if error_filepaths:
            error_filepaths.sort()
            st.error(f"{len(error_filepaths)} unwanted files found: ")
            with st.expander("Unwanted files:"):
                st.warning("  \n".join(error_filepaths))
            st.stop()

        if classif_annot_type != 'CSV file':
            # this is for image classification "Label by folder name"
            if not img_names:
                st.warning("No image uploaded")
                logger.info("No image uploaded ")
                st.stop()
            if return_annotations:
                return [], img_filepaths
            return img_filepaths

        if not req_file_idx or not img_names:
            if not req_file_idx:
                st.warning(f"No {filetype_name} file uploaded")
                logger.info(f"No {filetype_name} file uploaded")
            if not img_names:
                st.warning("No image uploaded")
                logger.info("No image uploaded ")
            st.stop()

        if len(req_file_idx) > 1:
            required_filetype = required_filetype.upper()
            st.error(f"""{len(req_file_idx)} {required_filetype} files found.
            You should only upload a single {required_filetype} file.""")
            st.stop()
        # get the required CSV or JSON file while removing it from the image filepaths
        annotation_filepath = filepaths.pop(req_file_idx[0])
        annotation_filebytes = extract_one_to_bytes(
            uploaded_archive, annotation_filepath)

        # ***************** Checking CSV file data *****************
        if required_filetype == '.csv':
            df = pd.read_csv(BytesIO(annotation_filebytes), dtype=str)

            check_required_csv_columns(df)

            if len(df) != len(filepaths):
                st.error(f"The CSV file has {len(df)} rows, which "
                         f"is not the same as the number of uploaded images: {len(filepaths)}")
                st.stop()

            # Currently NOT checking for backslash/frontslash now to allow
            # all kinds of inputs ,and use only the filenames of the images
            # to parse the annotations
            # check_invalid_paths(df)

            # get the image names in the CSV file
            annot_img_names = df.iloc[:, 0].apply(
                lambda x: os.path.basename(x)).tolist()

            # if len(df.iloc[:, 0].unique()) != len(df):
            if len(set(annot_img_names)) != len(annot_img_names):
                txt = ("Duplicated image filenames found, which would be confusing "
                       "to interpret! Please make sure the image names are unique.")
                st.error(txt)
                logger.error(txt)
                st.stop()

        # ***************** Checking JSON file data *****************
        elif required_filetype == '.json':
            coco_json = json.loads(annotation_filebytes)

            check_coco_json_keys(coco_json)

            annot_img_names = [os.path.basename(d['file_name'])
                               for d in coco_json['images']]
            annot_img_ids = set(
                [d['image_id'] for d in coco_json['annotations']])
            total_images = len(filepaths)
            error_txt = None
            if len(annot_img_ids) != total_images:
                error_txt = (
                    f"Every annotation should have its own valid image ID. "
                    f"But found {len(annot_img_ids)} image IDs in the COCO JSON "
                    f"annotations but you have uploaded {total_images} images.")
            if len(annot_img_names) != total_images:
                error_txt = (
                    f"Every image should have its own image ID. "
                    f"But found {len(annot_img_names)} image data in the COCO JSON "
                    f"but you have uploaded {total_images} images.")
            if error_txt is not None:
                st.error(error_txt)
                logger.error(error_txt)
                st.stop()

        # ************ Checking both CSV and JSON file for unknown image names ************
        unknown_annot_img = set(annot_img_names).difference(img_names)
        unknown_img_names = set(img_names).difference(annot_img_names)
        if len(unknown_annot_img) != 0 or len(unknown_img_names) != 0:
            if len(unknown_annot_img) != 0:
                unknown_annot_img = sorted(unknown_annot_img)
                st.error(f"""Every image should have its own annotation.
                But found {len(unknown_annot_img)} unknown image name(s) in the
                {filetype_name} file without associated image.""")
                with st.expander(f"List of the unknown names found in {filetype_name} file:"):
                    st.warning("  \n".join(unknown_annot_img))
            if len(unknown_img_names) != 0:
                unknown_img_names = sorted(unknown_img_names)
                st.error(f"""Every image should have its own annotation.
                But found {len(unknown_img_names)} image file(s) without associated
                annotation in the {filetype_name} file.""")
                with st.expander("List of the unknown images:"):
                    st.warning("  \n".join(unknown_img_names))
            st.stop()

        # ***************** Finalizing function *****************
        self.annotation_files = [annotation_filepath]
        if return_annotations:
            # return the CSV or JSON file in a List, and also the remaining image filepaths
            return self.annotation_files, img_filepaths
        return img_filepaths

    def parse_annotation_files(
            self, deployment_type: str, image_paths: List[str] = None, **kwargs) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Parse the uploaded annotation files based on the `deployment_type` and
        yield the image name and annotation result in Label Studio JSON result format
        to save in database annotations table"""
        if not image_paths:
            # get from dataset attr which might contain the paths
            image_paths = self.dataset
        if deployment_type == 'Object Detection with Bounding Boxes':
            return self.parse_bbox_annotations()
        elif deployment_type == 'Image Classification':
            return self.parse_classification_annotations(image_paths, **kwargs)
        elif deployment_type == 'Semantic Segmentation with Polygons':
            return self.parse_segmentation_annotations()

    def parse_bbox_annotations(self) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Parse the uploaded XML files and yield the image name and annotation result in
        Label Studio JSON result format"""
        for xml_filepath in self.annotation_files:
            # xml_str = xml_file.read().decode()
            full_path = self.archive_dir / xml_filepath
            try:
                image_name, annot_result = convert_xml2_ls(full_path)
            except Exception as e:
                st.error(f'Error parsing XML file "{xml_filepath}" with error: {e}  \n'
                         'Please try checking your annotation file(s) again before uploading.')
                logger.error("Error parsing XML file")
                # delete the invalid dataset
                self.delete_dataset(self.id)
                st.stop()
            # take only filename without extension, as explained above
            image_name = os.path.splitext(image_name)[0]
            yield image_name, annot_result

    def parse_classification_annotations(
            self, image_paths: List[str] = None,
            classif_annot_type: str = 'CSV file') -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Parse the uploaded CSV and yield the image name and annotation result in
        Label Studio JSON result format"""
        if classif_annot_type == 'CSV file':
            # only one CSV file after validation is passed
            csv_path = self.archive_dir / self.annotation_files[0]
            df = pd.read_csv(csv_path, dtype=str)
            for img_path, label in zip(df[CSV_IMAGE_COL], df[CSV_LABEL_COL]):
                img_name = os.path.basename(img_path)
                result = create_classification_result(label)
                yield img_name, result
        else:
            for p in image_paths:
                # img_name = os.path.basename(p)
                # get the label from folder name
                label = os.path.basename(os.path.dirname(p))
                result = create_classification_result(label)
                yield p, result

    def parse_segmentation_annotations(self) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Parse the uploaded JSON file and yield the image name and annotation result in
        Label Studio JSON result format"""
        # only one COCO JSON file after validation is passed
        json_path = self.archive_dir / self.annotation_files[0]
        with open(json_path) as f:
            coco_json = json.load(f)

        # with keys: width, height, file_name
        image_id2info = {}
        for image_dict in coco_json['images']:
            img_id = image_dict.pop("id")
            image_id2info[img_id] = image_dict

        cat_id2name = {}
        for category_dict in coco_json['categories']:
            cat_id = category_dict.pop("id")
            cat_id2name[cat_id] = category_dict['name']

        first = True
        for annot in coco_json['annotations']:
            annot_id = annot['id']
            img_id = annot['image_id']
            img_info = image_id2info.get(img_id)
            if img_info is None:
                txt = (
                    f'Unable to find the corresponding image info for image ID {img_id} for '
                    f'annotation ID {annot_id}. Each annotation must have its associated '
                    'image ID in the "image" values. Please check your COCO JSON file.')
                st.error(txt)
                logger.error(txt)
                # delete the invalid dataset
                self.delete_dataset(self.id)
                st.stop()
            relative_fpath = img_info['file_name']
            # filename = os.path.basename(relative_fpath)

            if first:
                first = False
                # take the first img_id from the COCO json file
                # sometimes could be 0, sometimes could be 1
                prev_img_id = img_id
                result = []
                prev_fpath = relative_fpath

            if prev_img_id != img_id:
                # generate the previous image filename and all the annotation results
                # for the image of prev_img_id
                logger.debug(f"Reached new image ID {img_id}, "
                             f"yielding result for previous img_id {prev_img_id}")
                yield prev_fpath, result
                # then reset the result list and update for new image_id
                result = []
                prev_img_id = img_id
                prev_fpath = relative_fpath

            original_width = img_info['width']
            original_height = img_info['height']

            try:
                points = convert_coco2ls_points(
                    annot['segmentation'],
                    original_width,
                    original_height
                )
            except Exception as e:
                st.error(f"Error converting COCO segmentation values to Label Studio "
                         f"scaled values for annotation ID {annot_id} with error: {e}"
                         f"  \nPlease try checking your "
                         "COCO JSON file again before uploading.")
                # delete the invalid dataset
                self.delete_dataset(self.id)
                st.stop()

            label = cat_id2name[annot['category_id']]

            current_annot_result = {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "points": points,
                    "polygonlabels": [label]
                },
                "id": annot_id,
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels"
            }
            result.append(current_annot_result)
        # yield the result for the final image
        logger.debug(f"Yielding result for final image {relative_fpath}")
        yield relative_fpath, result

    def insert_dataset(self):
        insert_dataset_SQL = """
                                INSERT INTO public.dataset (
                                    name,
                                    description,
                                    dataset_size,
                                    filetype_id)
                                VALUES (
                                    %s,
                                    %s,
                                    %s,
                                    
                                    (SELECT ft.id from public.filetype ft where ft.name = %s))
                                RETURNING id;
                            """
        self.dataset_size = len(self.dataset)
        insert_dataset_vars = [self.name, self.desc,
                               self.dataset_size, self.filetype]
        self.id = db_fetchone(
            insert_dataset_SQL, conn, insert_dataset_vars).id
        return self.id

    @staticmethod
    def reset_new_dataset_page():
        """Method to reset all widgets and attributes in the New Dataset Page when changing pages
        """

        reset_camera_and_ports()

        new_dataset_attributes = [
            "new_dataset", "is_labeled", "dataset_chosen"]

        reset_page_attributes(new_dataset_attributes)


class Dataset(BaseDataset):
    def __init__(self, dataset, data_name_list: Dict = {}) -> None:
        super(). __init__(dataset.ID)
        self.id = dataset.ID
        self.name = dataset.Name
        self.desc = dataset.Description
        self.dataset_size = dataset.Dataset_Size
        self.dataset_path = self.get_dataset_path(self.name)  # NOTE
        # self.dataset_path = dataset.Dataset_Path
        self.filetype = dataset.File_Type
        self.data_name_list = {}

    def get_data_name_list(self, data_name_list_full: Dict):
        """Obtain list of data in the dataset 
            - Iterative glob through the dataset directory
            - Obtain filename using pathlib.Path(<'filepath/*'>).name

        Returns:
            Dict[dict]: Dataset name as key to a List of data in the dataset directory
        """
        self.dataset_path = self.get_dataset_path(self.name)
        try:
            # IF dataset info already exist and len of data same as number of files in folder -> get from Dict
            if data_name_list_full.get(self.name) and (len(data_name_list_full.get(self.name))) == len([file for file in Path(self.dataset_path).iterdir() if file.is_file()]):

                self.data_name_list = data_name_list_full.get(self.name)

            else:
                data_name_list_full = self.glob_folder_data_list(
                    data_name_list_full)

        except AttributeError as e:
            logger.error(f"{e}: NoneType error for data_name_list dictionary")
            data_name_list_full = {}

        return data_name_list_full

    def glob_folder_data_list(self, data_name_list_full: Dict) -> Dict[List[str], Any]:
        """#### Get data info for data table:
            - id: Data Name
            - filetype: Data filetype (Image, Video,Audio, Text)
            - created: Date of modification in the filesystem

        Args:
            data_name_list_full (Dict): Existing list of data info

        Returns:
            Dict[List[str, Any]]:   Dictionary with dataset name as key and List of data info as value
        """
        self.dataset_path = self.get_dataset_path(self.name)
        logger.info(self.dataset_path)
        logger.info(self.name)
        if self.dataset_path:

            # dataset_path = Path(self.dataset_path) / "./*"
            dataset_path = Path(self.dataset_path)

            data_info_tmp = []

            # i need
            # {'id':data_name,'filetype':self.filetype,'created_at':os.stat().st_mtime}

            # Glob through dataset directory
            # for data_path in iglob(str(dataset_path)):
            for data_path in dataset_path.iterdir():
                if data_path.is_file():
                    data_info = {}

                    logger.info(f"Listing files in {data_path}......")
                    data_info['id'] = Path(data_path).name
                    data_info['filetype'] = self.filetype

                    # Get File Modified Time
                    data_modified_time_epoch = os.stat(str(data_path)).st_mtime
                    data_modified_time = datetime.fromtimestamp(data_modified_time_epoch
                                                                ).strftime('%Y-%m-%d')
                    data_info['created'] = data_modified_time
                    data_info_tmp.append(data_info)

                data_name_list_full[self.name] = data_info_tmp
                self.data_name_list = data_info_tmp

            return data_name_list_full

    def update_title_desc(self, new_name: str, new_desc: str):
        update_title_desc_SQL = """
                                    UPDATE
                                        public.dataset
                                    SET
                                        name = %s,
                                        description = %s
                                    WHERE
                                        id = %s
                                    RETURNING name,description;
        
                                    """
        update_title_desc_vars = [new_name, new_desc, self.id]
        new_title_desc_return = db_fetchone(
            update_title_desc_SQL, conn, update_title_desc_vars)
        logger.info(f"Updating title and desc {new_title_desc_return}")

        self.name, self.desc = new_title_desc_return if new_title_desc_return else (
            None, None)

    def update_dataset_path(self, new_dataset_name: str):
        # new_directory_name = get_directory_name(new_dataset_name)
        # new_dataset_path = DATASET_DIR / str(new_directory_name)
        new_dataset_path = self.get_dataset_path(new_dataset_name)

        # Rename dataset folder
        try:
            old_dataset_path = Path(self.dataset_path)
            old_dataset_path.rename(Path(new_dataset_path))
            logger.info(f'Renamed dataset path to {new_dataset_path}')
        except Exception as e:
            logger.error(f'{e}: Could not rename file')

    def update_dataset(self, archive_dir: Path):
        """Wrapper function to update existing dataset
        """
        # save added data into file-directory
        return self.save_dataset(archive_dir)

    def update_pipeline(self, success_place) -> int:
        """ Pipeline to update dataset

        Args:
            success_place (EmptyMixin): Placeholder to display dataset update progress and info

        Returns:
            int: Return append_data_flag as 0 to leave *data_upload_module*
        """

        if self.update_dataset():

            success_place.success(
                f"Successfully appended **{self.name}** dataset")

            sleep(1)

            success_place.empty()

            self.update_dataset_size()

            logger.info(
                f"Successfully updated **{self.name}** size in database")

            append_data_flag = 0

        else:
            success_place.error(
                f"Failed to append **{self.name}** dataset")
            append_data_flag = 1

        return append_data_flag

    def display_data_media_attributes(self, data_info: str, data_raw: Image.Image, filename: str = None, placeholder=None):
        if placeholder:
            placeholder = placeholder
        else:
            placeholder = st.empty()

        if data_info:
            data_name = data_info['id'] if data_info['id'] else Path(
                data_raw.filename).name
            created = data_info['created'] if data_info['created'] else ""
            mimetype = get_mime(data_name)

            if not data_raw:
                self.dataset_path = self.get_dataset_path(self.name)
                filepath = self.dataset_path / data_name

            try:
                filetype = data_info['filetype']
            except:
                filetype = str(Path(mimetype).parent)

            if isinstance(filetype, str):
                filetype = FileTypes.from_string(filetype)
            # Image
            if filetype == FileTypes.Image:
                image_width, image_height = get_image_size(data_raw)
                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Width: {image_width}
                - Height: {image_height}
                - MIME type: {mimetype}
                """
            # video
            elif filetype == FileTypes.Video:
                props = get_video_properties(filepath)

                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Codec:{props['codec_name']}
                - Width: {props['width']}
                - Height: {props['height']}
                - Duration: {float(props['duration']):.2f}s
                - Frame rate: {props['avg_frame_rate']}
                - Frame count: {props['nb_frames']}
                - MIME type: {mimetype}
                """
            # Audio
            elif filetype == FileTypes.Audio:

                props = get_audio_properties(filepath)

                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - Codec:{props['codec_long_name']}
                - Channel Layout: {props['channel_layout']}
                - Channels: {props['channels']}
                - Duration: {float(props['duration']):.2f}s
                - Bit rate: {(float(props['bit_rate'])/1000)}kbps
                - Sampling rate: {(float(props['sample_rate'])/1000):.2f}kHz
                - MIME type: {mimetype}
                """
            # Text
            elif filetype == FileTypes.Text:
                display_attributes = f"""
                ### **DATA**
                - Name: {data_name}
                - Created: {created}
                - Dataset: {self.name}
                ___
                ### **MEDIA ATTRIBUTES**
                - MIME type: {mimetype}
                """
            if placeholder:
                with placeholder.container():
                    st.info(display_attributes)
            else:
                st.write("### \n")
                st.info(display_attributes)

    def load_dataset(self):

        self.dataset_path = self.get_dataset_path(self.name)

        for file in Path(self.dataset_path).iterdir():
            if file.is_file():
                pass

    @staticmethod
    def get_filetype_enumerator(data_name: Union[str, Path, UploadedFile]) -> IntEnum:
        """Query enumerated constants for FileTypes IntEnum class

        Args:
            data_name (Union[str, Path, UploadedFile]): Name of data with extensions (eg. IMG_20210831.png)

        Returns:
            (IntEnum): FileTypes constant
        """

        filetype = get_filetype(data_name).capitalize()
        filetype = FileTypes.from_string(filetype)

        return filetype

    @staticmethod
    def query_related_projects(
            dataset_id: int, deployment_type: str, is_labelled: bool = False
    ) -> List[NamedTuple]:
        sql_query = """
            SELECT p.id, p.name
            FROM project p
                    LEFT JOIN deployment_type dt ON p.deployment_id = dt.id
            WHERE dt.name = %s
                AND p.id IN (
                    SELECT DISTINCT project_id
                    FROM task
                    WHERE dataset_id = %s
                        AND is_labelled = %s
            );
        """
        query_vars = [deployment_type, dataset_id, is_labelled]
        projects = db_fetchall(sql_query, conn, query_vars)
        return projects

# **************************** IMPORTANT ****************************
# ************************ CANNOT CACHE !!!!!*************************
# Will throw ValueError for selectbox dataset_sel because of session state (BUG)


def query_dataset_list() -> List[NamedTuple]:
    """Query list of dataset from DB, Column Names: TRUE

    Returns:
        NamedTuple: List of datasets
    """
    query_dataset_SQL = """
        SELECT
            id AS "ID",
            name AS "Name",
            dataset_size AS "Dataset Size",
            (SELECT ft.name AS "File Type" from public.filetype ft where ft.id = d.filetype_id),
            updated_at AS "Date/Time",
            description AS "Description"
            
        FROM
            public.dataset d
        ORDER BY id ASC;"""

    datasets, column_names = db_fetchall(
        query_dataset_SQL, conn, fetch_col_name=True)
    logger.info("Querying dataset from database......")
    dataset_tmp = []
    if datasets:
        for dataset in datasets:

            # convert datetime with TZ to (2021-07-30 12:12:12) format
            converted_datetime = dataset.Date_Time.strftime(
                '%Y-%m-%d %H:%M:%S')

            dataset = dataset._replace(
                Date_Time=converted_datetime)
            dataset_tmp.append(dataset)

        # self.dataset_list = dataset_tmp
    else:
        dataset_tmp = []

    return dataset_tmp, column_names

# **************************** IMPORTANT ****************************
# ************************ CANNOT CACHE !!!!!*************************
# Will throw ValueError for selectbox dataset_sel because of session state (BUG)


def get_dataset_name_list(dataset_list: List[NamedTuple]) -> Dict[str, NamedTuple]:
    """Generate Dictionary of namedtuple

    Args:
        dataset_list (List[Namedtuple]): Query from database

    Returns:
        Dict: Dictionary of dataset_name -> dataset's NamedTuple record from database
    """

    # dataset_name_list = {}  # list of dataset name for selectbox
    dataset_dict = {}  # use to store named tuples as value to dataset name as key

    if dataset_list:
        for dataset in dataset_list:
            # DEPRECATED -> dataset info can be accessed through namedtuple of dataset_dict
            # dataset_name_list[dataset.Name] = dataset.ID
            dataset_dict[dataset.Name] = dataset
        logger.info("Generating list of dataset names and ID......")
    return dataset_dict


def generate_image_name(original_name: str, all_img_names: Set[str]) -> str:
    """
    Provide `all_img_names` to be able to check whether the image names 
    exist within the `all_img_names` or not to generate another new one if exists.
    """
    lowercase_ori_name = original_name.lower()
    lowercase_ori_name_split = lowercase_ori_name.split(".")
    lowercase_extension = lowercase_ori_name_split[-1]
    lowercase_img_name = "".join(lowercase_ori_name_split[:-1])
    # must compare with lowercase as filenames are usually case-insensitive in
    # in most platforms, e.g. in Windows
    allowed_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'

    def get_code():
        return get_random_string(length=5, allowed_chars=allowed_chars)
    new_img_name = f"{lowercase_img_name}_{get_code()}.{lowercase_extension}"
    if all_img_names:
        while new_img_name in all_img_names:
            # to ensure unique names
            new_img_name = f"{lowercase_img_name}_{get_code()}.{lowercase_extension}"
        all_img_names.add(new_img_name)
    return new_img_name


def save_single_image(
        img_path: str, dataset_path: Path,
        all_img_names: Set[str] = None, verbose: bool = False) -> Tuple[bool, str, str]:
    """Save a single image from `img_path` to `dataset_path`. A random string
    is always prepended to the image filename before saving to `dataset_path`. 

    Provide `all_img_names` to be able to check whether the image names 
    exist within the `all_img_names` or not to generate another new one if exists.

    Returns Tuple[bool, str ,str] for `(success, original_image_name, new_image_name)`
    """
    ori_img_name = os.path.basename(img_path)
    new_img_name = generate_image_name(ori_img_name, all_img_names)

    save_path = dataset_path / new_img_name
    success = False
    try:
        cv2.imread(img_path)  # test reading image
        shutil.move(img_path, save_path)
    except ValueError as e:
        logger.error(
            f"{e}: Could not resolve output format for '{ori_img_name}'")
    except OSError as e:
        logger.error(
            f"{e}: Failed to create file '{new_img_name}'. File may exist or contain partial data")
    except Exception as e:
        logger.error(f"Unknown error occurred with '{img_path}': {e}")
    else:
        success = True
        if verbose:
            relative_dataset_path = Path(
                dataset_path).relative_to(BASE_DATA_DIR)
            logger.debug(
                f"Successfully stored '{new_img_name}' in '{relative_dataset_path}'")
    return success, ori_img_name, new_img_name


def load_image(image_path: str, opencv_flag: bool = True) -> Union[Image.Image, np.ndarray]:
    """Loads image via OpenCV into Numpy arrays or through PIL into Image class object

    Args:
        image_path (str): Path to image
        opencv_flag (bool, optional): True to process by OpenCV. Defaults to True.

    Returns:
        Union[Image.Image, np.ndarray]: Image object
    """
    if opencv_flag:
        image_path = str(image_path)
        image = cv2.imread(image_path)

    else:

        image = Image.open(image_path)

    return image


def data_url_encoder(filetype: IntEnum, data_path: Union[str, Path]) -> str:
    """Generate Data URL

    Args:
        filetype (IntEnum): FileTypes IntEnum class constants
        data_path (Union[str, Path]): Path to data

    Returns:
        str: String of base64 encoded data url
    """
    # REMOVED: `data_object` args is not needed anymore
    if filetype == FileTypes.Image:
        image_name = os.path.basename(data_path)

        logger.debug("Encoding image with base64 to display with Label Studio")
        with open(data_path, "rb") as f:
            b64code = b64encode(f.read()).decode('utf-8')

        # if isinstance(data_object, np.ndarray):
        #     image_name = Path(data_path).name

        #     logger.debug(f"Encoding image into bytes: {str(image_name)}")
        #     extension = Path(image_name).suffix
        #     _, buffer = cv2.imencode(extension, data_object)
        #     logger.debug("Done enconding into bytes")

        #     logger.debug("Start B64 Encoding")

        #     b64code = b64encode(buffer).decode('utf-8')
        #     logger.debug("Done B64 encoding")

        # elif isinstance(data_object, Image.Image):
        #     img_byte = BytesIO()
        #     image_name = Path(data_object.filename).name  # use Path().name
        #     logger.debug(f"Encoding image into bytes: {str(image_name)}")
        #     data_object.save(img_byte, format=data_object.format)
        #     logger.debug("Done enconding into bytes")

        #     logger.debug("Start B64 Encoding")
        #     bb = img_byte.getvalue()
        #     b64code = b64encode(bb).decode('utf-8')
        #     logger.debug("Done B64 encoding")

        mime = guess_type(image_name)[0]
        logger.debug(f"{image_name} ; {mime}")
        data_url = f"data:{mime};base64,{b64code}"
        logger.debug("Data url generated")

        return data_url

    elif filetype == FileTypes.Video:
        pass
    elif filetype == FileTypes.Audio:
        pass
    elif filetype == FileTypes.Text:
        pass


def get_latest_captured_image_path() -> Tuple[Path, int]:
    """Returns the latest image path and image number based on existing 
    images in the directory."""

    def get_image_num(image_path: Path):
        return int(image_path.name.split('_')[0])

    existing_captured = sorted(
        CAPTURED_IMAGES_DIR.rglob('*.png'),
        key=get_image_num,
        reverse=True)
    if not existing_captured:
        image_num = 1
    else:
        image_num = get_image_num(existing_captured[0]) + 1

    filename = f'{image_num}_{get_random_string(8)}.png'
    save_path = CAPTURED_IMAGES_DIR / filename
    return save_path, image_num


def find_image_path(
        image_paths: List[str], query_image_name: str, deployment_type: str) -> Union[None, str]:
    """Find the image_path for the `query_image_name` out of the `image_paths`.

    Returns None if not found."""
    for p in image_paths:
        if deployment_type == 'Object Detection with Bounding Boxes':
            # taking only the filename without extension to consider the case of
            #  Label Studio exported XML files without any file extension
            image_name = os.path.basename(os.path.splitext(p)[0])
        else:
            image_name = os.path.basename(p)
        if query_image_name == image_name:
            return p
    logger.debug(f"Image file not found for {query_image_name}")


def main():
    print("Hi")


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
