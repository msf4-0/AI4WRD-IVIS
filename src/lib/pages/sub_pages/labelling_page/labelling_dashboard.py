"""
Title: Labelling Dashboard
Date: 19/8/2021
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

import os
import shutil
import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from time import sleep
from typing import List

import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


from annotation.annotation_management import (LabellingPagination,
                                              reset_editor_page)
from core.utils.log import logger  # logger
from data_editor.data_labelling import editor
from data_editor.editor_config import editor_config
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from project.project_management import (ExistingProjectPagination, Project,
                                        ProjectPermission)
from annotation.annotation_management import Task
from data_manager.data_table_component.data_table import data_table

# NOTE Temp
from user.user_management import User
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
# conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
place = {}

chdir_root()  # change to root directory

# **************** DATA TABLE COLUMN CONFIG *********************************************************
all_task_columns = [
    {
        'field': "id",
        'headerName': "ID",
        'headerAlign': "center",
        'align': "center",
        'flex': 70,
        'hideSortIcons': True,

    },
    {
        'field': "Task Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 200,
        'hideSortIcons': False,
    },
    {
        'field': "Created By",
        'headerAlign': "center",
        'align': "center",
        'flex': 130,
        'hideSortIcons': True,
    },
    {
        'field': "Dataset Name",
        'headerAlign': "center",
        'align': "center",
        'flex': 150,
        'hideSortIcons': False,
    },
    {
        'field': "Is Labelled",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },
    {
        'field': "Skipped",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': True,
        'type': 'boolean',
    },
    {
        'field': "Date/Time",
        'headerAlign': "center",
        'align': "center",
        'flex': 100,
        'hideSortIcons': False,
        'type': 'date',
    },

]

# **************** DATA TABLE COLUMN CONFIG *********************************************************

# ************************* TO LABELLING CALLBACK ***********************************


def to_labelling_editor_page(section_enum: IntEnum):
    labelling_section_data_table_dict = {
        LabellingPagination.AllTask: 'all_task_table_key',
        LabellingPagination.Labelled: 'labelled_task_table_key',
        LabellingPagination.Queue: 'task_queue_table_key'
    }

    labelling_section_data_table = labelling_section_data_table_dict[section_enum]

    # Get state value from the respective Data Table
    session_state.data_selection = deepcopy(
        session_state[labelling_section_data_table])
    # Change pagination to Editor
    session_state.labelling_pagination = LabellingPagination.Editor
    # Reset selection at Data Table
    if labelling_section_data_table in session_state:
        del session_state[labelling_section_data_table]
    if "new_annotation_flag" not in session_state:
        session_state.new_annotation_flag = 0
    else:
        session_state.new_annotation_flag = 0
    logger.info(
        f"Data selected, entering editor for data {session_state.data_selection} for {labelling_section_data_table}")
    st.experimental_rerun()

# ************************* TO LABELLING CALLBACK ***********************************


def no_labelled(all_task, labelled_task_dict, task_queue_dict):

    # >>>> TASK QUEUE TABLE >>>>>>>>>>>>>>>>>>>>>>
    # >>>> All task table placeholders

    task_queue_table_col1, task_queue_table_col2 = st.columns([3, 0.75])
    length_of_queue = len(task_queue_dict)
    total_task_length = len(all_task)

    task_queue_table_col1.write(f"### Task Queue")
    task_queue_table_col2.write(
        f"### Total remaining data: {length_of_queue}/{total_task_length}")
    data_table(task_queue_dict, all_task_columns,
               checkbox=False, key='task_queue_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.Queue,))


def labelled_table(all_task, labelled_task_dict, task_queue_dict):

    # >>>> LABELLED TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    labelled_task_table_col1, labelled_task_table_col2 = st.columns([3, 0.75])

    length_of_labelled = len(labelled_task_dict)

    labelled_task_table_col1.write(f"### Labelled Task")
    labelled_task_table_col2.write(
        f"### Total labelled data: {length_of_labelled}/{len(all_task)}")
    data_table(labelled_task_dict, all_task_columns,
               checkbox=False, key='labelled_task_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.Labelled,))


def all_task_table(all_task, labelled_task_dict, task_queue_dict):

    # >>>> ALL TASK TABLE >>>>>>>>>>>>>>>>>>>>>>

    # >>>> All task table placeholders
    (all_task_table_col1, all_task_table_col2,
     all_task_table_col3, all_task_table_col4,
     all_task_table_col5) = st.columns([2, 0.3, 0.5, 0.5, 0.5])
    _, all_task_table_bottom_col1, all_task_table_bottom_col2 = st.columns([
                                                                           3, 0.75, 0.75])

    all_task_table_col1.write(f"### All Task")
    all_task_table_col2.write(f"### Total data: {len(all_task)}")
    n_skipped_data = 0
    for task in labelled_task_dict:
        if task['Skipped']:
            n_skipped_data += 1

    n_remaining = len(all_task) - len(labelled_task_dict)

    all_task_table_col3.write(
        f"### Total labelled data: {len(labelled_task_dict)}")
    all_task_table_col4.write(
        f"### Total skipped data: {n_skipped_data}")
    all_task_table_col5.write(
        f"### Total remaining data: {n_remaining}")

    data_table(all_task, all_task_columns,
               checkbox=False, key='all_task_table_key', on_change=to_labelling_editor_page, args=(LabellingPagination.AllTask,))


def export_section():
    if 'zipfile_path' not in session_state:
        # initialize the path to the zipfile for images & annotations
        session_state['zipfile_path'] = None

    st.subheader("Export data")
    table_place = st.empty()
    export_labels_col, download_task_col, _ = st.columns([1, 1, 3])
    archive_success_message = st.empty()

    with table_place.container():
        st.markdown(
            "#### You can export dataset in one of the following formats:")
        converter = session_state.project.editor.get_labelstudio_converter()
        format_df, name2enum_str = session_state.project.editor.get_supported_format_info(
            converter)
        st.table(format_df)

        info_col, _ = st.columns([2, 1])
        with info_col:
            depl_type = session_state.project.deployment_type
            if depl_type == "Image Classification":
                st.info("""📝 Recommended to choose CSV format, which will also copy your
                images into individual folders for each class.""")
            elif depl_type == "Object Detection with Bounding Boxes":
                st.info("""📝 Recommended to choose Pascal VOC XML format, which is one of
                the most common formats used for many object detection training algorithms,
                also the format used for our TensorFlow Object Detection training in this
                application.""")
            elif depl_type == "Semantic Segmentation with Polygons":
                st.info("""📝 Recommended to choose COCO JSON format, which is the most common
                format used for image segmentation tasks, also the format used for our training
                in this application. Choosing COCO format will also export both the original 
                and the mask images.""")

            st.selectbox("Select your choice of format to export the labeled tasks:",
                        options=format_df['Format'], key='export_format')

    def download_export_tasks():
        with st.spinner("Creating the zipfile, this may take awhile depending on your dataset size..."):
            # zipfile_path = session_state.project.download_tasks(
            #     return_target_path=True)
            format_enum_str = name2enum_str[session_state.export_format]
            zipfile_path = session_state.project.download_tasks(
                converter=converter,
                export_format=format_enum_str,
                return_original_path=True)
            session_state['zipfile_path'] = zipfile_path
            logger.info(f"Zipfile created at: {zipfile_path}")

    with export_labels_col:
        btn_export = st.button("Export Tasks", key='export_labels_button')

    if btn_export:
        # use this instead of callback to show the message below the button
        download_export_tasks()
        st.experimental_rerun()

    def reset_zipfile_state():
        # clear out the `download_button` after the user has clicked it
        os.remove(session_state['zipfile_path'])
        session_state['zipfile_path'] = None

    with download_task_col:
        zipfile_path = session_state.get('zipfile_path')
        downloaded = False
        if zipfile_path is not None and zipfile_path.exists():
            with st.spinner("Creating the Zipfile button to download ... This may take awhile ..."):
                file_name = f"images-annotations-{session_state.export_format}.zip"
                with open(zipfile_path, "rb") as fp:
                    downloaded = st.download_button(
                        label="Download Zipfile",
                        data=fp,
                        file_name=file_name,
                        mime="application/zip",
                        key="download_tasks_btn",
                        on_click=reset_zipfile_state,
                    )

    if downloaded:
        # - commenting out this line in case we are only deploying for local machine,
        # - this is to show message to the user about the "Downloads" folder path
        # archive_success_message.success(
        #     f"Zipfile created successfully at **{session_state['zipfile_path']}**")
        archive_success_message.success(
            f"Zipfile created successfully, you may download it by pressing the 'Download ZIP' button")


def index(RELEASE=True):
    # ****************** TEST ******************************
    if not RELEASE:
        logger.debug("At Labelling INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        # for Anson: 4 for TFOD, 9 for img classif, 30 for segmentation
        project_id_tmp = 109
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************

    st.write(f"## **Labelling Section:**")

    # ************COLUMN PLACEHOLDERS *****************************************************
    # labelling_section_clusters_button_col,_,start_labelling_button_col=st.columns([1,3,1])
    filter_msg_col, _, all_task_button_col, _, labelled_task_button_col, _, queue_button_col, _ = st.columns(
        [0.5, 0.5, 1.5, 0.5, 2, 0.5, 2, 3])
    filter_msg_col.markdown("**Filter:**")

    action_msg_col, _, start_labelling_button_col, _, edit_labeling_config_col, _, export_section_col, _ = st.columns(
        [0.5, 0.5, 1.5, 0.5, 2, 0.5, 2, 3])
    action_msg_col.markdown("**Action:**")
    # ************COLUMN PLACEHOLDERS *****************************************************

    labelling_page = {
        LabellingPagination.AllTask: all_task_table,
        LabellingPagination.Labelled: labelled_table,
        LabellingPagination.Queue: no_labelled,
        LabellingPagination.Editor: editor,
        LabellingPagination.EditorConfig: editor_config,
        LabellingPagination.Performance: None,
        LabellingPagination.Export: export_section
    }

    # >>>> INSTANTIATE LABELLING PAGINATION
    if 'labelling_pagination' not in session_state:
        session_state.labelling_pagination = LabellingPagination.AllTask
    if 'data_selection' not in session_state:
        # state store for Data Table
        session_state.data_selection = []
    if "show_next_unlabeled" not in session_state:
        # a flag to decide whether to show next unlabeled data
        session_state.show_next_unlabeled = False

    # >>>> PAGINATION BUTTONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def to_labelling_section(section_name: IntEnum, show_next: bool = False):
        session_state.labelling_pagination = section_name
        reset_editor_page()
        logger.debug(f"Show next unlabeled: {show_next}")
        session_state.show_next_unlabeled = show_next

    with all_task_button_col:
        st.button("All Task", key='all_task_button', on_click=to_labelling_section, args=(
            LabellingPagination.AllTask,))

    with labelled_task_button_col:
        st.button("Labelled Task", key='labelled_task_button',
                  on_click=to_labelling_section, args=(LabellingPagination.Labelled,))

    with queue_button_col:
        st.button("Queue", key='queue_button', on_click=to_labelling_section, args=(
            LabellingPagination.Queue,))

    with start_labelling_button_col:
        # only show next when clicking "Start Labelling" button, instead of
        # clicking on the data_table directly
        st.button("Start Labelling", key='start_labelling_button',
                  on_click=to_labelling_section, args=(LabellingPagination.Editor, True))

    with edit_labeling_config_col:
        st.button("Edit Editor/Labeling Config", key='edit_labelling_config_button',
                  on_click=to_labelling_section, args=(LabellingPagination.EditorConfig,))

    with export_section_col:
        st.button("Export", key='export_section_button',
                  on_click=to_labelling_section, args=(LabellingPagination.Export,))

    # >>>> PAGINATION BUTTONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.debug(f"Navigator: {session_state.labelling_pagination}")

    if session_state.labelling_pagination == LabellingPagination.Editor:
        labelling_page[session_state.labelling_pagination](
            session_state.data_selection)

    elif session_state.labelling_pagination == LabellingPagination.EditorConfig:
        labelling_page[session_state.labelling_pagination](
            session_state.project)

    elif session_state.labelling_pagination == LabellingPagination.Export:
        labelling_page[session_state.labelling_pagination]()

    else:
        session_state.show_next_unlabeled = False
        if 'all_task' not in session_state:
            all_task, _ = Task.query_all_task(
                session_state.project.id,
                return_dict=True,
                for_data_table=True)
            session_state.all_task = all_task
        all_task = session_state.all_task
        labelled_task_dict = Task.get_labelled_task(all_task, True)
        task_queue_dict = Task.get_labelled_task(all_task, False)
        labelling_page[session_state.labelling_pagination](
            all_task, labelled_task_dict, task_queue_dict)


if __name__ == "__main__":
    # DEFINE wide layout for debugging when running this script/page directly
    # layout = 'wide'
    # st.set_page_config(page_title="Integrated Vision Inspection System",
    #                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        index(RELEASE=False)

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
