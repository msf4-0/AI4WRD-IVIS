"""
Title: Data Uploader
Date: 9/8/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from time import sleep, perf_counter
from typing import Union, Dict
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from copy import deepcopy
from core.webcam import webcam_webrtc
from core.utils.helper import check_filetype
from data_manager.database_manager import init_connection
from data_manager.dataset_management import Dataset, query_dataset_list, get_dataset_name_list
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# >>>> Variable Declaration >>>>
place = {}  # PLACEHOLDER

# TODO #44


def data_uploader(dataset: Dataset, key: str = None):

    chdir_root()  # change to root directory

    # ******** SESSION STATE ********
    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        session_state.data_source = "File Upload 📂"
    # ******** SESSION STATE ********


# TODO
    # >>>>>>>> New Dataset Upload >>>>>>>>
    # else:
    # pass
    # if 'webcam_flag' not in session_state:
    #     session_state.webcam_flag = False
    #     session_state.file_upload_flag = False
    #     # session_state.img1=True

    st.write("## __Dataset Upload:__")
    data_source_options = ["Webcam 📷", "File Upload 📂"]
    # col1, col2 = st.columns(2)

    data_source = st.radio(
        "Data Source", options=data_source_options, key="data_source_radio")
    data_source = data_source_options.index(data_source)

    dataset_size_string = f"- ### Number of datas: **{dataset.dataset_size}**"
    dataset_filesize_string = f"- ### Total size of data: **{(dataset.calc_total_filesize()):.2f} MB**"

    st.markdown(" ____ ")
    # TODO: #15 Webcam integration
    # >>>> WEBCAM >>>>
    if data_source == 0:
        webcam_webrtc.app_loopback()

    # >>>> FILE UPLOAD >>>>
    # TODO #24 Add other filetypes based on filetype table
    # Done #24

    elif data_source == 1:

        # uploaded_files_multi = st.file_uploader(
        #     label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget", on_change=check_filetype_category, args=(place,))
        uploaded_files_multi = st.file_uploader(
            label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget")

        place["upload"] = st.empty()

        # ******** INFO for FILE FORMAT **************************************
        st.write("#### File Format Infomation")
        st.write("\n")

        file_format_info = """
        1. Image: .jpg, .png, .jpeg
        2. Video: .mp4, .mpeg
        3. Audio: .wav, .mp3, .m4a
        4. Text: .txt, .csv
        """
        st.info(file_format_info)

        if uploaded_files_multi and (len(session_state.upload_widget) != 0):

            # >>>> CHECK FILETYPE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            check_filetype(
                uploaded_files_multi, dataset, place)

            dataset.dataset = deepcopy(uploaded_files_multi)

            dataset.dataset_size = len(
                uploaded_files_multi)  # length of uploaded files
            log_info("Enter data exist")
        else:
            log_info("Enter else")
            dataset.filetype = None
            dataset.dataset_size = 0  # length of uploaded files
            dataset.dataset = None
            
        dataset_size_string = f"- ### Number of datas: **{dataset.dataset_size}**"
        dataset_filesize_string = f"- ### Total size of data: **{(dataset.calc_total_filesize()):.2f} MB**"

        st.markdown(" ____ ")

        # ******************* DATASET METRICS **************************************
        dataset_size_place = st.empty()
        dataset_size_place.write(dataset_size_string)

        dataset_filesize_place = st.empty()
        dataset_filesize_place.write(dataset_filesize_string)
        dataset_size_place.write(dataset_size_string)
        dataset_filesize_place.write(dataset_filesize_string)
        st.markdown(" ____ ")
        # ******************* DATASET METRICS **************************************

    # Placeholder for WARNING messages of File Upload widget

    # with st.expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_files_multi:
    #         imgcol1.image(uploaded_files_multi[1])

    # TODO: KIV

    # col1, col2, col3 = st.columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>> SUBMISSION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    success_place = st.empty()
    field = [dataset.dataset]

    submit_button = st.button("Submit", key="submit")
    st.write(field)

    if submit_button:
        dataset.has_submitted = dataset.check_if_field_empty(
            field, field_placeholder=place, keys=['upload'])

        if dataset.has_submitted:
            # st.success(""" Successfully created new dataset: {0}.
            #                 """.format(dataset.name))

            if dataset.save_dataset():

                success_place.success(
                    f"Successfully created **{dataset.name}** dataset")

                if dataset.insert_dataset():

                    success_place.success(
                        f"Successfully stored **{dataset.name}** dataset information in database")

                    # reset NewDataset class object
                    dataset = NewDataset(
                        get_random_string(length=8))

                else:
                    st.error(
                        f"Failed to stored **{dataset.name}** dataset information in database")
            else:
                st.error(
                    f"Failed to created **{dataset.name}** dataset")

    st.write(vars(dataset))
    # for img in dataset.dataset:
    #     st.image(img)


def main():
    TEST_FLAG = True

    # ****************** TEST ******************************
    if TEST_FLAG:

        place = {}
        place['test'] = st.empty()
        existing_dataset, _ = query_dataset_list()
        dataset_dict = get_dataset_name_list(existing_dataset)
        if 'dataset' not in session_state:
            session_state.dataset = Dataset(dataset_dict['My Third Dataset'])
        if 'append_data_flag' not in session_state:
            session_state.append_data_flag = 0
        st.write(session_state.append_data_flag)
        # with place['test'].expander(label='Append dataset', expanded=False):
        if session_state.append_data_flag == 0:
            with place['test'].container():
                st.write("Normal")
                st.table([1, 2, 3, 4, 5])

        elif session_state.append_data_flag == 1:
            with place['test'].container():

                data_uploader(session_state.dataset)

        submit = st.button("Test", key="testing")
        if submit:
            with place['test'].container():
                st.success("Can")
                st.success("Yea")

        def flag_1():
            session_state.append_data_flag = 1

        def flag_0():
            session_state.append_data_flag = 0
            if 'upload_widget' in session_state:
                del session_state.upload_widget

        st.button("Flag 0", key='flag0', on_click=flag_0)
        st.button("Flag 1", key='flag1', on_click=flag_1)


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
