"""
Title: New Dataset Page
Date: 7/7/2021
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

import gc
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter, sleep

import cv2
import streamlit as st
from annotation.annotation_management import Annotations, NewTask, Task
from core.utils.code_generator import get_random_string
from core.utils.file_handler import (
    check_image_files,
    extract_archive,
    list_files_in_archived,
)
from core.utils.helper import (
    list_available_cameras,
    reset_camera,
    reset_camera_and_ports,
)
from core.utils.log import logger
from data_manager.database_manager import init_connection
from data_manager.dataset_management import (
    NewDataset,
    find_image_path,
    get_latest_captured_image_path,
    save_single_image,
)
from humanize import naturalsize
from imutils.video.webcamvideostream import WebcamVideoStream
from natsort import os_sorted
from path_desc import TEMP_DIR, chdir_root
from project.project_management import (
    NewProject,
    Project,
    ProjectPagination,
    ProjectPermission,
)
from stqdm import stqdm
from streamlit import cli as stcli
from streamlit import session_state
from user.user_management import User

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

# # DEFINE wide page layout for debugging on this page directly
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration >>>>
# new_dataset = {}  # store
place = {}


def new_dataset(RELEASE=True, conn=None, is_new_project: bool = True, is_existing_dataset: bool = False):
    """Function for the page of creating new dataset or adding more images to an existing
    project dataset.

    `is_new_project`: A flag of creating new dataset for a new project, to ensure we
    can get the default editor_config template labels to remove properly.

    `is_existing_dataset`: A flag of updating an existing dataset, i.e. to add more data to
    an existing project dataset (which was selected in existing_project_dashboard.py).

    `session_state.is_labeled`: A flag to tell whether the user chooses to upload a
    labeled dataset, then validation will also be required by using
    `NewDataset.validate_labeled_data()`
    """
    if not conn:
        conn = init_connection(**st.secrets["postgres"])

    chdir_root()  # change to root directory

    # ******** DEBUGGING ********
    # NOTE: If debugging for inserting uploaded annotations, you need to select
    #  an existing project_id
    if not RELEASE:
        # debugging upload dataset
        session_state.is_labeled = True

        if not session_state.is_labeled and ("new_project" not in session_state):
            session_state.new_project = NewProject(get_random_string(length=8))
            # session_state.new_project.deployment_type = "Object Detection with Bounding Boxes"
            # session_state.new_project.deployment_type = "Image Classification"
            session_state.new_project.deployment_type = "Semantic Segmentation with Polygons"
        if session_state.is_labeled and ('project' not in session_state):
            project_id = 30
            logger.debug(f"""Entering Project ID {project_id} for debugging
            uploading labeled dataset""")
            session_state.project = Project(project_id)
    # ******** DEBUGGING ********

    # ******** SESSION STATE ********
    if "new_dataset" not in session_state:
        # set random dataset ID before getting actual from Database
        logger.debug("Enter new dataset")
        session_state.new_dataset = NewDataset(get_random_string(length=8))
    if 'user' not in session_state:
        session_state.user = User(1)
    if 'is_labeled' not in session_state:
        session_state.is_labeled = False
    logger.debug(f"{session_state.is_labeled = }")
    if session_state.is_labeled and ('project' not in session_state):
        # initialize project with the inserted project, note that this only works
        #  after the new_project has been stored in database
        project_id = session_state.new_project.id
        del session_state['new_project']
        session_state.project = Project(project_id)
        logger.info(f"Project ID {project_id} initialized from NewProject"
                    "to accept uploaded labeled dataset")

    dataset: NewDataset = session_state.new_dataset
    # ******** SESSION STATE ********

    if 'project' in session_state:
        project: Project = session_state.project
        deployment_type = project.deployment_type
    elif 'new_project' in session_state:
        deployment_type = session_state.new_project.deployment_type

    def clean_archive_dir():
        # remove the unneeded extracted archive dir contents
        with st.spinner("Removing the unwanted extracted files ..."):
            shutil.rmtree(dataset.archive_dir)
            logger.info(
                "Removed temporary directory for extracted contents")

    if is_existing_dataset:
        # session_state.dataset_chosen should be obtained from existing_project_dashboard
        dataset_info = project.dataset_dict[session_state.dataset_chosen]
        # set the info to be equal to new_dataset to make things easier
        dataset.id = dataset_info.ID
        dataset.name = dataset_info.Name
        dataset.desc = dataset_info.Description

    # >>>>>>>> New Dataset INFO >>>>>>>>
    # Page title
    if session_state.is_labeled or is_existing_dataset:
        st.write(f"# __Deployment Type: {deployment_type}__")
    if not is_existing_dataset:
        if session_state.is_labeled:
            st.write("## __Upload Labeled Dataset__")
            logger.info("Upload labeled dataset")
        else:
            st.write("# __Add New Dataset__")
    st.markdown("___")

    # right-align the dataset ID relative to the page
    _, id_right = st.columns([3, 1])
    id_right.write(
        f"### __Dataset ID:__ {dataset.id}")

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        input_name = session_state.name.strip()
        # NOTE: MUST BE LOWERCASE TO AVOID REPLACING EXISTING DIRECTORIES
        # as directory name is case-insensitive
        lower_input_name = input_name.lower()
        context = {'column_name': 'name',
                   'value': lower_input_name}
        if lower_input_name:
            if dataset.check_if_exists(context, conn):
                dataset.name = None
                field_placeholder['name'].error(
                    f"Dataset name used. Please enter a new name")
                sleep(1)
                logger.error(f"Dataset name used. Please enter a new name")
            else:
                # save the user's input one but check with lowercase one
                dataset.name = input_name
                logger.info(f"Dataset name fresh and ready to rumble")

    # >>>>>>> DATASET INFORMATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not is_existing_dataset:
        outercol1.write("## __Dataset Information :__")
        outercol2.text_input(
            "Dataset Title", key="name", help="Enter the name of the dataset", on_change=check_if_name_exist, args=(place, conn,))
        place["name"] = outercol2.empty()

        # **** Dataset Description (Optional) ****
        description = outercol2.text_area(
            "Description (Optional)", key="desc", help="Enter the description of the dataset")
        if description:
            dataset.desc = description
    else:
        st.write("## __Current Project Dataset Information :__")
        st.markdown(f"**Dataset name:** {dataset.name}")
        st.markdown(
            f"**Dataset description:** {dataset.desc}")

        st.markdown("## __Adding data to existing project dataset__")
        labeled = st.radio(
            "Select type of dataset to upload", options=('Not Labeled', 'Labeled'),
            index=0, key='labeled'
        )
        session_state.is_labeled = True if labeled == 'Labeled' else False

    # <<<<<<<< New Dataset INFO <<<<<<<<

    # >>>>>>>> New Dataset Upload >>>>>>>>

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])

    outercol1.write("## __Dataset Upload:__")

    if not session_state.is_labeled:
        data_source_options = ["Webcam 📷", "File Upload 📂"]
        # col1, col2 = st.columns(2)

        data_source = outercol2.radio(
            "Data Source", options=data_source_options, key="data_source_radio",
            index=1)
        data_source = data_source_options.index(data_source)
    else:
        data_source = 1

    outercol1, outercol2, outercol3 = st.columns([1.5, 2, 2])

    # TODO: #15 Webcam integration
    # >>>> WEBCAM >>>>
    if data_source == 0:
        # NOTE: not using streamlit-webrtc for now
        # webcam_webrtc.app_loopback()

        if 'working_ports' not in session_state:
            session_state.working_ports = []

        with outercol2:

            camera_type = st.radio("Select type of camera", ('USB Camera', 'IP Camera'),
                                   key='input_camera_type', on_change=reset_camera)
            if camera_type == 'USB Camera':
                num_check_ports = st.number_input(
                    "Number of ports to check for", 5, 99, 5,
                    key='input_num_check_ports',
                    help="""Number of ports to check whether they are working or not. If you are using
                    Docker on Linux, you need to specify a number at least as high as the highest number
                    of the camera *dev path*. For example, **9** if the largest *dev path* is:
                    **/dev/video9**.""")
                st.button("Refresh camera ports", key='btn_refresh',
                          on_click=reset_camera_and_ports)
                if not session_state.working_ports:
                    with st.spinner("Checking available camera ports ..."):
                        _, working_ports = list_available_cameras(
                            num_check_ports)
                        session_state.working_ports = working_ports.copy()
                if not session_state.working_ports:
                    st.error("No available camera port found.")
                    st.stop()
                camera_port = st.radio("Select a camera port", session_state.working_ports,
                                       key='camera_port')
                cam_key = f'camera'
                video_source = camera_port
            else:
                ip_cam_address = st.text_input(
                    "Enter the IP address", key='ip_cam_address')
                with st.expander("Notes about IP Camera Address"):
                    st.markdown(
                        """IP camera address could start with *http* or *rtsp*.
                        Most of the IP cameras have a username and password to access
                        the video. In such case, the credentials have to be provided
                        in the streaming URL as follow:
                        **rtsp://username:password@192.168.1.64/1**""")
                if not ip_cam_address:
                    st.warning("Please enter an IP address")
                    st.stop()
                # ip camera needs to include 'ip' to avoid release it in reset_camera()
                cam_key = f'camera_ip'
                video_source = ip_cam_address

            camera_btn_place = st.empty()

            if not session_state.get(cam_key):
                if camera_btn_place.button(
                    "Start camera", key='btn_start_cam',
                        help='Start camera before start capturing images'):
                    with st.spinner(f"Loading up camera ..."):
                        try:
                            session_state[cam_key] = WebcamVideoStream(
                                src=video_source).start()
                            if session_state[cam_key].read() is None:
                                raise Exception(
                                    "Video source is not valid")
                        except Exception as e:
                            st.error(
                                f"Unable to read from video source {video_source}")
                            logger.error(
                                f"Unable to read from video source {video_source}: {e}")
                            reset_camera()
                            st.stop()
                else:
                    st.stop()

            camera_btn_place.button(
                "Stop camera", key='btn_stop_cam', on_click=reset_camera)

        stream = session_state[cam_key].stream
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(stream.get(cv2.CAP_PROP_FPS))
        logger.info(
            f"Webcam properties: {width = }, {height = }, {fps_input = }")

        with outercol1:
            st.markdown(f"### Webcam info:  \n**Width**: {width}, "
                        f"**Height**: {height}, **FPS**: {fps_input}")

            save_path, image_num = get_latest_captured_image_path()
            save_dir = save_path.parent
            if not save_dir.exists():
                os.makedirs(save_dir)
            if os.environ.get("DOCKERCOMPOSE"):
                # this save directory is the local Linux directory to be able to access the
                # captured_images directory mounted into the Docker container
                # NOTE: this path depends on the volume name used in the docker-compose file
                # and this path is only for Linux PC because currently camera device
                # is not supported on Docker created on Windows WSL
                save_dir = Path(r"/var/lib/docker/volumes/integrated-vision-inspection-system_app-data/_data",
                                save_dir.name)
            st.markdown(
                f"Images will be saved in this directory: *{save_dir}*")
            display_width = st.slider(
                "Select width of image to resize for display",
                35, 1000, 640, 5, key='display_width',
                help="This does not affect the size of the captured image as it depends on the camera.")

            is_limiting = st.checkbox(
                "Limit images captured per second", value=True)
            if is_limiting:
                img_per_sec = st.slider(
                    "Max images per second",
                    0.1, 30.0, 2.0, 0.1, format='%.1f', key='cps',
                    help="Note that this is not exactly precise but very close.")

        with outercol2:
            video_place = st.empty()
            start_capture = st.checkbox("Start capturing", key='start_capture')

        start_time = perf_counter()
        total_new_imgs = 0
        while True:
            frame = session_state[cam_key].read()
            video_place.image(frame, channels='BGR', width=display_width)

            if start_capture:
                elapsed_secs = perf_counter() - start_time
                if is_limiting and (total_new_imgs / elapsed_secs) > img_per_sec:
                    continue
                # note that opencv only accepts str and not Path
                cv2.imwrite(str(save_path), frame)
                total_new_imgs += 1
                image_num += 1
                save_path = save_path.with_name(
                    f'{image_num}_{get_random_string(8)}.png')
                # sleep to limit save rate a little bit
                sleep(0.1)

    # >>>> FILE UPLOAD >>>>
    # TODO #24 Add other filetypes based on filetype table
    # Done #24

    elif data_source == 1:

        # uploaded_files_multi = outercol2.file_uploader(
        #     label="Upload Image", type=['jpg', "png", "jpeg", "mp4", "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"], accept_multiple_files=True, key="upload_widget", on_change=check_filetype_category, args=(place,))
        # allowed_types = ['jpg', "png", "jpeg", "mp4",
        #                  "mpeg", "wav", "mp3", "m4a", "txt", "csv", "tsv"]
        # allowed_types = ['jpg', "png", "jpeg", "txt", "csv", "xml", "json"]
        allowed_types = ['zip', 'tar.gz', 'tar.xz', 'tar.bz2']
        uploaded_archive = outercol2.file_uploader(
            label="Upload Image", type=allowed_types, accept_multiple_files=False, key="upload_widget")
        # outercol2.info("""NOTE: When you are uploading a lot of files at once, the
        # Streamlit app may look like it's stuck but please wait for it to finish uploading.""")
        place["upload"] = outercol2.empty()
        # ******** INFO for FILE FORMAT **************************************
        with outercol1.expander("File Format Infomation", expanded=True):
            # not using these formats for our application
            # 2. Video: .mp4, .mpeg
            # 3. Audio: .wav, .mp3, .m4a
            # 4. Text: .txt, .csv
            file_format_info = """
            #### Image Format:
            Image: .jpg, .png, .jpeg

            **NOTE**: Only a single zipfile or tarfile with all the required files inside can be accepted.
            """
            st.info(file_format_info)
            if session_state.is_labeled:
                st.info(
                    """
                    #### Compatible Annotation Format:  \n
                    - **Object detection** should have one XML file for each uploaded image.  \n
                    **Filenames** should be **unique** for the best results.  \n
                    - **Image classification** can have two types:  \n
                    1. *CSV file*: should only have images and only one CSV file,
                    the CSV file must contain an *'image'* column for the image path
                    (but the image filenames will be extracted), and another *'label'* column
                    containing the labels for the images.  \n
                    Note that the filenames **must be unique**.  \n
                    2. *Label by folder names*: Each image is labeled by the folder name they
                    reside in. E.g. *cat1.jpg* image is in a folder named as *cat*, this
                    image will be labeled as *cat*  \n
                    - **Image segmentation** should only have images and only one COCO JSON file.  \n
                    **Filenames** should also be **unique** for the best results.
                    """)

        # default to CSV file for everything else
        classif_annot_type = 'CSV file'
        if session_state.is_labeled and deployment_type == 'Image Classification':
            with outercol2:
                options = ("CSV file", "Label by folder name")
                classif_annot_type = st.radio(
                    "Select annotation type", options)

        if uploaded_archive:
            # st.write(len(uploaded_archive))
            # st.write(uploaded_archive)

            # NOTE: once you closed the Streamlit UploadedFile (archive file in this case),
            # note that the contents **will be empty** the next time you open it again,
            # unless you created a deepcopy beforehand. Or you may choose to manually
            # open and close the archive file without context manager. Alternatively,
            # you can run `uploaded_archive.seek(0)` every time you want to reopen
            # the archive file
            with outercol2:
                with st.spinner("Getting archive content names and size ..."):
                    filepaths, content_size = list_files_in_archived(
                        archived_filepath=uploaded_archive.name,
                        file_object=uploaded_archive,
                        return_content_size=True,
                        skip_dir=True)
                # check_filetype(
                #     uploaded_files_multi, dataset, place)

            # dataset.dataset = image_names

            # length of uploaded files
            num_files = len(filepaths)
        else:
            content_size = 0
            num_files = 0
            # dataset.dataset = []

        with outercol3:
            dataset_size_string = f"- ### Number of datas: **{num_files}**"
            # dataset_filesize_string = f"- ### Total size of data: **{naturalsize(value=dataset.calc_total_filesize(),format='%.2f')}**"
            dataset_filesize_string = ("- ### Total size of data: "
                                       f"**{naturalsize(value=content_size, format='%.2f')}**")
            st.markdown(" ____ ")
            st.write(dataset_size_string)
            st.write(dataset_filesize_string)
            st.markdown(" ____ ")

    # Placeholder for WARNING messages of File Upload widget

    # with st.expander("Data Viewer", expanded=False):
    #     imgcol1, imgcol2, imgcol3 = st.columns(3)
    #     imgcol1.checkbox("img1", key="img1")
    #     for image in uploaded_archive:
    #         imgcol1.image(uploaded_archive[1])

    # TODO: KIV

    # col1, col2, col3 = st.columns([1, 1, 7])
    # webcam_button = col1.button(
    #     "Webcam 📷", key="webcam_button", on_click=update_webcam_flag)
    # file_upload_button = col2.button(
    #     "File Upload 📂", key="file_upload_button", on_click=update_file_uploader_flag)

    # <<<<<<<< New Dataset Upload <<<<<<<<
    # ******************************** SUBMISSION *************************************************
    success_place = st.empty()

    # st.write("context")
    # st.write(context)
    submit_col1, submit_col2 = st.columns([3, 0.5])
    with submit_col2:
        # using a placeholder for button to be able to clear it out later
        button_place = st.empty()
        submit_button = button_place.button("Submit", key="submit")

    if submit_button:
        if not is_existing_dataset:
            input_name = session_state.name.strip()
        else:
            input_name = dataset.name
        lower_input_name = input_name.lower()

        # check lowercase name, but insert user input name later
        context = {'name': lower_input_name,
                   'upload': session_state.upload_widget}
        if is_existing_dataset:
            # don't need to check for dataset name for existing dataset
            del context['name']

        dataset.has_submitted = dataset.check_if_field_empty(
            context, field_placeholder=place, name_key='name')
        if not dataset.has_submitted:
            st.stop()
        # save the user input name, not the lowercase one
        dataset.name = input_name

        with outercol2:
            if session_state.is_labeled:
                with st.spinner("Checking uploaded dataset and annotations ..."):
                    logger.info(
                        "Checking uploaded dataset and annotations")
                    start = perf_counter()
                    image_paths = dataset.validate_labeled_data(
                        uploaded_archive, filepaths, deployment_type,
                        classif_annot_type=classif_annot_type)
                    time_elapsed = perf_counter() - start
                    logger.info(
                        f"Done. [{time_elapsed:.4f} seconds]")
                    st.success("""🎉 Annotations are verified to be compatible and
                    in the correct format.""")
            else:
                with st.spinner("Checking uploaded images ..."):
                    logger.info("Checking uploaded images")
                    image_paths = check_image_files(filepaths)

        logger.info(f"Found {len(image_paths)} images in the archive.")
        # sort to always make it maintain the same order every time if possible
        image_paths.sort()
        dataset.dataset = image_paths

        # create a temporary directory for extracting the archive contents
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)
        dataset.archive_dir = TEMP_DIR

        with outercol3:
            with st.spinner("Extracting uploaded archive contents ..."):
                extract_archive(TEMP_DIR, file_object=uploaded_archive)

        # For labeled dataset, don't save the images to disk because it's
        # easier to store together with annotations later
        save_images_to_disk = False if session_state.is_labeled else True
        _, new_img_names, error_img_paths = dataset.save_dataset(
            dataset.archive_dir, save_images_to_disk=save_images_to_disk)
        if error_img_paths:
            st.error("Failed to save dataset!")
            with st.expander("The following images cannot be read:"):
                st.markdown("\n  ".join(error_img_paths))
            dataset.delete_dataset(dataset.id)
            st.stop()

        txt = f"Successfully created and saved **{dataset.name}** dataset"
        success_place.success(txt)
        logger.info(txt)

        txt = ""
        if not is_existing_dataset:
            # NEW dataset
            if dataset.insert_dataset():
                txt = f"Successfully stored **{dataset.name}** dataset information in database"
                logger.info(txt)
                success_place.success(txt)
            else:
                txt = "Unable to update/insert dataset information"
                logger.error(txt)
                success_place.error(txt)
                st.stop()

            # We need to insert the NEW project_dataset here after the dataset
            # has been stored in database
            if isinstance(session_state.get('project'), Project):
                with st.spinner("Inserting the project dataset ..."):
                    # add the uploaded dataset as the dataset_chosen, to
                    # allow the insert_project_dataset to work
                    dataset_name = dataset.name
                    dataset_chosen = [dataset_name]
                    # if uploading labeled dataset, skip inserting task here because
                    # we will insert them later to easily rename and link images
                    insert_task = False if session_state.is_labeled else True
                    project.insert_project_dataset(
                        dataset_chosen, insert_task=insert_task)
                    logger.info(f"Inserted project dataset '{dataset_name}' for "
                                f"Project {project.id} into project_dataset table")
                    # must refresh all the dataset details
                    project.refresh_project_details()

        # NOTE: stop here if not uploading any labeled dataset
        if not session_state.is_labeled:
            # NOTE: currently existing dataset in this module is assumed
            #  to be linked with a project.
            # so insert the new image info into the `task` table for existing dataset,
            # only insert if it is not labeled, because images, tasks,
            # and annotations will be stored together later for labeled dataset
            if is_existing_dataset:
                # `os_sorted` is used to sort the files just like in file browser
                # to make the order of the files make more sense to the user
                # especially in the labelling pages ('data_labelling.py'
                # & 'labelling_dashboard.py')
                image_names = os_sorted(new_img_names)
                project.insert_new_project_task(
                    dataset.name,
                    dataset.id,
                    image_names=image_names)
                # update dataset size here before stopping
                if dataset.update_dataset_size():
                    txt = f"Successfully updated {dataset.name} dataset size information"
                    logger.info(txt)
                    project.refresh_project_details()
            txt = ("Successfully stored the new dataset in database "
                   "and local storage.")
            logger.info(txt)
            success_place.success(txt)
            with st.spinner("Cleaning temporary archive files ..."):
                clean_archive_dir()
            st.stop()

        # ******************** FOR LABELED DATASET **********************

        total_images = len(image_paths)
        logger.info(
            f"Found {total_images} images in the database.")
        filetype = dataset.filetype
        logger.info("Submitting uploaded annotations ...")

        start_t = perf_counter()
        error_imgs = []
        result_generator = dataset.parse_annotation_files(
            project.deployment_type,
            image_paths=image_paths,
            classif_annot_type=classif_annot_type
        )
        # use this to keep track of existing task names (i.e. image names)
        all_img_names = set(dataset.get_image_paths(
            dataset.name, return_names=True))
        message = "Inserting uploaded annotations into database"
        for relative_img_path, result in stqdm(result_generator, total=total_images,
                                               unit=filetype, desc=message):
            # start_task = perf_counter()
            full_image_path = os.path.join(
                dataset.archive_dir, relative_img_path)
            if not os.path.exists(full_image_path):
                error_txt = (
                    f"Image '{relative_img_path}' not found. Maybe the image "
                    "does not exist, or was removed "
                    "because unable to read the image. Skipping from "
                    "submitting to database.")
                if deployment_type == 'Object Detection with Bounding Boxes':
                    # check without file extension
                    image_name = os.path.basename(
                        os.path.splitext(relative_img_path)[0])
                else:
                    image_name = os.path.basename(relative_img_path)

                # attempt to find the image path
                image_fpath = find_image_path(
                    image_paths, image_name, deployment_type)
                if not image_fpath:
                    logger.error(error_txt)
                    error_imgs.append(relative_img_path)
                    continue

                full_image_path = os.path.join(
                    dataset.archive_dir, image_fpath)
                if not os.path.exists(full_image_path):
                    logger.error(error_txt)
                    error_imgs.append(relative_img_path)
                    continue

            success, ori_img_name, new_img_name = save_single_image(
                full_image_path, dataset.dataset_path, all_img_names)
            if not success:
                txt = f"Unable to read or save the image for: {relative_img_path}"
                logger.error(txt)
                st.error(txt)
                dataset.delete_dataset(dataset.id)
                st.stop()

            task_id = NewTask.insert_new_task(
                new_img_name, project.id, dataset.id)
            annot_id = Annotations.insert_annotation(
                result, session_state.user.id,
                project.id, task_id)
            Task.update_task(task_id, annot_id,
                             is_labelled=True, skipped=False)

            # time_elapsed = perf_counter() - start_task
            # logger.debug(
            # f"Annotation submitted successfully [{time_elapsed:.4f}s]")

        time_elapsed = perf_counter() - start_t
        logger.info("Done inserting all annotations for "
                    f"{total_images} images into database. "
                    f"Took {time_elapsed:.2f} seconds. "
                    f"Average {time_elapsed / total_images:.4f}s per image")

        if error_imgs:
            if len(error_imgs) == total_images:
                st.error("""All annotations were not saved successfully. Please check
                again for any errors with the images. Then please go back to project
                creation page and try again.""")
                dataset.delete_dataset(dataset.id)
                project.delete_project(project.id)
                NewProject.reset_new_project_page()
                NewDataset.reset_new_dataset_page()
                # also could be coming from project dashboard
                Project.reset_dashboard_page()
                session_state.project_pagination = ProjectPagination.Dashboard
                st.stop()
            txt = """NOTE: These images were not found/unreadable and
                thus skipped, but others are stored successfully in the
                database. You should check that the annotation file points
                to the correct image path. If you think it's safe to ignore, then
                you may now proceed to enter the current project."""
            with st.expander(txt):
                st.warning("  \n".join(error_imgs))
        else:
            success_place.success("""🎉 All images and annotations are successfully
            stored in database. You may now proceed to enter the current project.""")

        if is_existing_dataset:
            if dataset.update_dataset_size():
                txt = f"Successfully updated **{dataset.name}** dataset size information"
                logger.info(txt)

        with st.spinner("Updating project labels and editor configuration ..."):
            project.update_editor_config(
                is_new_project, refresh_project=True)

        # clear out the "Submit" button to avoid further interactions
        button_place.empty()

        def enter_project_cb():
            NewProject.reset_new_project_page()
            NewDataset.reset_new_dataset_page()
            # also could be coming from project dashboard
            Project.reset_dashboard_page()
            session_state.project_pagination = ProjectPagination.Existing
            session_state.project_status = ProjectPagination.Existing
            session_state.append_project_flag = ProjectPermission.ViewOnly

            logger.info(
                f"Entering Project {project.id}")
            gc.collect()

        st.button("Enter Project", key="btn_enter_project",
                  on_click=enter_project_cb)

        clean_archive_dir()

    # FOR DEBUGGING:
    # st.write("vars(dataset)")
    # st.write(vars(dataset))
    # from copy import deepcopy

    # project: Project = project
    # editor = deepcopy(project.editor)
    # existing_config_labels = editor.get_labels()
    # st.write("existing_config_labels")
    # st.write(existing_config_labels)
    # st.write(type(editor))
    # st.write("vars(editor)")
    # st.write(vars(editor))
    # st.write(editor.editor_config)
    # # st.write(editor.xml_doc.getElementsByTagName(
    # #     editor.parent_tagname))
    # st.write(editor.create_label('value', 'haha'))
    # st.write(editor.get_labels())


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        st.sidebar.markdown("Dummy sidebar")
        # initialise connection to Database
        conn = init_connection(**st.secrets["postgres"])

        new_dataset(RELEASE=False, conn=conn)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
