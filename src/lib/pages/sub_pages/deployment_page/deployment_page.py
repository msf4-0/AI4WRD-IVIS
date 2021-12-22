"""
Title: Deployment
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

from copy import deepcopy
from datetime import datetime
from functools import partial
import json
import os
import shutil
import sys
from time import perf_counter, sleep

import cv2
from imutils.video.webcamvideostream import WebcamVideoStream
from matplotlib import pyplot as plt
import numpy as np
from paho.mqtt.client import Client
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
from streamlit.report_thread import add_report_ctx
from project.project_management import Project

from user.user_management import User, UserRole

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

from path_desc import TEMP_DIR, chdir_root
from core.utils.log import logger
from data_manager.database_manager import init_connection
from machine_learning.visuals import create_class_colors, create_color_legend
from data_manager.dataset_management import Dataset
from deployment.deployment_management import DeploymentConfig, DeploymentPagination, DeploymentType, Deployment
from deployment.utils import MQTTConfig, MQTTTopics, create_csv_file_and_writer, image_from_buffer, image_to_bytes, get_mqtt_client, read_images_from_uploaded, reset_camera, reset_video_deployment, reset_camera_and_ports, reset_csv_file_and_writer, reset_record_and_vid_writer
from core.utils.helper import Timer, get_all_timezones, get_now_string, list_available_cameras, save_image
from dobot_arm_demo import main as dobot_demo

# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


def kpi_format(text: str):
    return f"<h1 style='text-align: center; color:red;'>{text}</h1>"


def index(RELEASE=True):
    if 'deployment' not in session_state:
        st.warning("You have not deployed any model yet.")
        st.warning(
            "Please go to the Model Selection page and deploy a model first.")
        st.stop()

    if 'deployed' not in session_state:
        session_state.deployed = False
    if 'camera' not in session_state:
        session_state.camera = None

        # use this in the video loop to refresh the page once to show widget changes
        session_state.refresh = False

    if 'working_ports' not in session_state:
        session_state.working_ports = None
    if 'deployment_conf' not in session_state:
        # to store the config in case the user needs to go to another page during deployment
        session_state.deployment_conf = DeploymentConfig()
    if 'client' not in session_state:
        # NOTE: using project ID as the client ID for now
        session_state.client = get_mqtt_client(
            str(session_state.project.id))
        session_state.client_connected = False
        # to check whether video callbacks have been added to MQTT client
        session_state.added_video_cbs = False
        session_state.mqtt_conf = MQTTConfig()

        # to check whether is publishing results through MQTT
        session_state.publishing = True
    if 'mqtt_recv_frame' not in session_state:
        # to check whether any image/frame is received through MQTT
        session_state.mqtt_recv_frame = None

    deploy_conf: DeploymentConfig = session_state.deployment_conf
    client: Client = session_state.client
    conf: MQTTConfig = session_state.mqtt_conf
    topics: MQTTTopics = session_state.mqtt_conf.topics

    project: Project = session_state.project
    deployment: Deployment = session_state.deployment
    user: User = session_state.user
    DEPLOYMENT_TYPE = deployment.deployment_type

    def reset_image_idx():
        """Reset index of image chosen to avoid exceeding the max number of images
        on widget changes"""
        session_state.image_idx = 0

    def image_recv_frame_cb(client, userdata, msg):
        # logger.debug("FRAME RECEIVED, REFRESHING")
        session_state.mqtt_recv_frame = msg.payload
        # only refresh for image type
        session_state.refresh = True

    def recv_frame_cb(client, userdata, msg):
        # not refreshing here to speed up video deployment speed
        session_state.mqtt_recv_frame = msg.payload

    def subscribe_topics(resubscribe: bool = False):
        for attr, topic in topics.__dict__.items():
            if attr.startswith('publish'):
                # skip these topics because we are only publishing things to these topics
                # instead of subscribing to them to wait for input
                continue
            if resubscribe:
                client.unsubscribe(topic)
            client.subscribe(topic, qos=conf.qos)

    def update_mqtt_qos():
        # take from the widget's state and save to our mqtt_conf
        logger.info(f"Updated QoS level from {conf.qos} to "
                    f"{session_state.mqtt_qos}")
        conf.qos = session_state.mqtt_qos

        subscribe_topics(resubscribe=True)

    def end_deployment():
        Deployment.reset_deployment_page()
        session_state.deployment_pagination = DeploymentPagination.Models

    def update_deploy_conf(conf_attr: str):
        """Update deployment config on any change of the widgets.

        NOTE: `conf_attr` must exist in the `session_state` (usually is the widget's state)
        and must be the same with the `DeploymentConfig` attribute's name."""
        val = session_state[conf_attr]
        logger.debug(f"Updated deploy_conf: {conf_attr} = {val}")
        setattr(deploy_conf, conf_attr, val)

    # connect MQTT broker and set up callbacks
    if not session_state.client_connected:
        logger.debug(f"{conf = }")
        with st.spinner("Connecting to MQTT broker ..."):
            try:
                client.connect(conf.broker, port=conf.port)
            except Exception as e:
                st.error("Error connecting to MQTT broker")
                if os.getenv('DEBUG', '1') == '1':
                    st.exception(e)
                logger.error(
                    f"Error connecting to MQTT broker {conf.broker}: {e}")
                st.stop()

            sleep(2)  # Wait for connection setup to complete
            logger.info("MQTT client connected successfully to "
                        f"{conf.broker} on port {conf.port}")

            subscribe_topics()

            # add callbacks for image input type first because the input type
            # defaults to "Image"
            client.message_callback_add(
                topics.recv_frame, image_recv_frame_cb)

            client.loop_start()

            # need to add this to avoid Missing ReportContext error
            # https://github.com/streamlit/streamlit/issues/1326
            add_report_ctx(client._thread)

            session_state.client_connected = True

    deploy_status_col = st.container()

    with deploy_status_col:
        st.subheader("Deployment Status")

        deploy_status_place = st.empty()
        deploy_status_place.info("**Status**: Not deployed. Please upload an image/video "
                                 "or use video camera or MQTT and click **\"Deploy Model\"**.")

        st.warning("**NOTE**: Please do not simply refresh the page without ending "
                   "deployment or errors could occur.")

        deploy_btn_place = st.empty()

        st.button(
            "End Deployment", key='btn_stop_image_deploy',
            on_click=end_deployment,
            help="This will end the deployment and reset the entire  \n"
            "deployment configuration. Please **make sure** to use this  \n"
            "button to stop deployment before proceeding to any other  \n"
            "page! Or use the **pause button** (only available for camera  \n"
            "input) if you want to pause the deployment to do any other  \n"
            "things without resetting, such as switching user, or viewing  \n"
            "the latest saved CSV file (only for video camera deployment).")
        st.markdown("___")

    if user.role <= UserRole.Developer1:
        # use this variable to know whether the user has access to edit deployment config
        has_access = True
    else:
        has_access = False
        st.info(f"NOTE: Your user role **{user.role.fullname}** "
                "does not have access to editing deployment configuration.")
        st.markdown("___")

    st.sidebar.subheader("Configuration")

    if has_access:
        def update_input_type_conf():
            reset_image_idx()
            session_state.mqtt_recv_frame = None
            client.message_callback_remove(topics.recv_frame)
            if session_state.input_type == 'Video':
                client.message_callback_add(
                    topics.recv_frame, recv_frame_cb)
            else:
                client.message_callback_add(
                    topics.recv_frame, image_recv_frame_cb)

            deploy_conf.input_type = session_state.input_type
        all_timezones = get_all_timezones()
        tz_idx = all_timezones.index(deploy_conf.timezone)
        st.sidebar.selectbox(
            "Local Timezone", all_timezones, index=tz_idx, key='timezone',
            help="Select your local timezone to have the correct time output in results.",
            on_change=update_deploy_conf, args=('timezone',))

        options = ('Image', 'Video')
        idx = options.index(deploy_conf.input_type)
        st.sidebar.radio(
            'Choose the Type of Input', options,
            index=idx, key='input_type',
            on_change=update_input_type_conf)
    else:
        st.info(f"Selected timezone: **{deploy_conf.timezone}**")
        st.markdown(f"**Input type**: {deploy_conf.input_type}")

    options_col, _ = st.columns(2)

    if DEPLOYMENT_TYPE == 'Image Classification':
        pipeline_kwargs = {}
    elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
        class_colors = create_class_colors(deployment.class_names)
        ignore_background = st.sidebar.checkbox(
            "Ignore background", value=deploy_conf.ignore_background,
            key='ignore_background',
            help="Ignore background class for visualization purposes.  \n"
            "Note that turning this on could significantly reduce the FPS.",
            on_change=update_deploy_conf, args=('ignore_background',))
        legend = create_color_legend(
            class_colors, bgr2rgb=False, ignore_background=ignore_background)
        st.sidebar.markdown("**Legend**")
        st.sidebar.image(legend)
        # convert to array
        class_colors = np.array(list(class_colors.values()),
                                dtype=np.uint8)
        pipeline_kwargs = {'class_colors': class_colors,
                           'ignore_background': ignore_background}
    else:
        if has_access:
            options_col.slider(
                "Confidence threshold:",
                min_value=0.1,
                max_value=0.99,
                value=deploy_conf.confidence_threshold,
                step=0.01,
                format='%.2f',
                key='confidence_threshold',
                help=("If a prediction's confidence score exceeds this threshold, "
                      "then it will be displayed, otherwise discarded."),
                on_change=update_deploy_conf, args=('confidence_threshold',)
            )
        else:
            options_col.markdown(
                f"**Confidence threshold**: {deploy_conf.confidence_threshold}")
        pipeline_kwargs = {'conf_threshold': deploy_conf.confidence_threshold}

    if deploy_conf.input_type == 'Image':
        image_type = st.sidebar.radio(
            "Select type of image",
            ("Image from project datasets", "Uploaded Image", "From MQTT"),
            key='select_image_type', on_change=reset_image_idx)
        if image_type == "Image from project datasets":
            project_datasets = project.data_name_list.keys()
            selected_dataset = st.sidebar.selectbox(
                "Select a dataset from project datasets",
                project_datasets, key='selected_dataset')

            project_image_names = project.data_name_list[selected_dataset]
            total_images = len(project_image_names)
            default_n = 10 if total_images >= 10 else total_images
            n_images = st.sidebar.number_input(
                "Select number of images to load",
                1, total_images, default_n, 5, key='n_images',
                help=(f"Total images in the project is **{total_images}**.  \n"
                      "Choose a lower value to reduce memory consumption."))
            project_image_names = project_image_names[:n_images]

            if 'image_idx' not in session_state:
                session_state.image_idx = 0

            filename = st.sidebar.selectbox(
                "Select a sample image from the project dataset",
                project_image_names, index=session_state.image_idx, key='filename')
            image_idx = project_image_names.index(filename)
            filename = project_image_names[image_idx]
            dataset_path = Dataset.get_dataset_path(selected_dataset)
            image_path = str(dataset_path / filename)

            def random_select():
                session_state.image_idx = np.random.randint(
                    len(project_image_names))

            st.sidebar.button("Randomly select one", key='btn_random_select',
                              on_click=random_select)

            st.markdown("**Selected image from project dataset**")
            img: np.ndarray = cv2.imread(image_path)
            # using this to cater to the case of multiple uploaded images
            imgs_info = ((img, filename),)
        elif image_type == "Uploaded Image":
            uploaded_imgs = st.sidebar.file_uploader(
                "Upload image(s)", type=['jpg', 'jpeg', 'png'],
                key='image_uploader_deploy', accept_multiple_files=True,
                help="Note that uploading too many images (more than 100) could be very "
                "taxing on the app.")
            if not uploaded_imgs:
                st.stop()

            imgs_info = read_images_from_uploaded(uploaded_imgs)
        else:
            def update_image_recv_topic():
                previous_topic = topics.recv_frame
                new_topic = session_state.image_recv_frame

                client.unsubscribe(previous_topic)
                client.message_callback_remove(previous_topic)

                topics.recv_frame = new_topic
                client.subscribe(new_topic)
                client.message_callback_add(new_topic, image_recv_frame_cb)
                logger.info(
                    f"Updated topic from {previous_topic} to {new_topic}")

            st.sidebar.markdown("___")
            st.sidebar.subheader("MQTT QoS")
            st.sidebar.radio(
                'MQTT QoS', (0, 1, 2), conf.qos, key='mqtt_qos',
                on_change=update_mqtt_qos)

            st.sidebar.subheader("MQTT Topic")
            st.sidebar.text_input(
                'Publish frame to our app', topics.recv_frame,
                key='image_recv_frame', help="Publish the image frame in bytes to this topic "
                "for MQTT input deployment", on_change=update_image_recv_topic)

            if not deploy_btn_place.checkbox(
                    "Deploy model for MQTT input", key='btn_deploy_mqtt_image'):
                st.stop()

            msg_place = st.empty()
            if not session_state.mqtt_recv_frame:
                logger.info("Waiting for image from MQTT ...")
                # this session_state is set to True in the recv_frame_cb() callback
                while not session_state.refresh:
                    msg_place.info(
                        "No frame received from MQTT. Waiting ...")
                    sleep(1)
                # reset the state and continue once received
                session_state.refresh = False
            msg_place.info("Image received **from MQTT**")
            logger.info("Reading the image from MQTT")
            img = image_from_buffer(session_state.mqtt_recv_frame)
            filename = 'Image from MQTT'
            # using this to cater to the case of multiple uploaded images
            imgs_info = ((img, filename),)

        if DEPLOYMENT_TYPE != 'Semantic Segmentation with Polygons':
            ori_image_width = img.shape[1]
            display_width = st.sidebar.slider(
                "Select width of image to resize for display",
                35, 1000, 500, 5, key='display_width')
            # help=f'Original image width is **{ori_image_width}**.')
            st.sidebar.markdown(f"Original image width: **{ori_image_width}**")

        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=True, **pipeline_kwargs)

        deploy_status_place.info("**Status**: Deployed for input images")

        if DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            with st.expander("Notes about deployment for semantic segmentation"):
                st.markdown(
                    """If this is an externally trained segmentation model, it might 
                    not work properly for making predictions for our app's implementation.
                    Please try with sample images or uploaded images first to see the results. 
                    If there is anything wrong, please do not proceed to video deployment.""")

        for img, filename in imgs_info:
            with st.spinner("Running inference ..."):
                try:
                    with Timer("Inference on image"):
                        inference_output = inference_pipeline(img)
                except Exception as e:
                    if os.getenv('DEBUG', '1') == '1':
                        st.exception(e)
                    logger.error(
                        f"Error running inference with the model: {e}")
                    st.error("""Error when trying to run inference with the model,
                        please check with Admin/Developer for debugging.""")
                    st.stop()
            if DEPLOYMENT_TYPE == 'Image Classification':
                pred_classname, y_proba = inference_output
                caption = (f"{filename}; "
                           f"Predicted: {pred_classname}; "
                           f"Score: {y_proba * 100:.1f}")
                st.image(img, channels='BGR',
                         width=display_width, caption=caption)
            elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
                drawn_mask_output, _ = inference_output
                # convert to RGB for visualizing with Matplotlib
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.subheader(filename)
                fig = plt.figure()
                plt.subplot(121)
                plt.title("Original Image")
                plt.imshow(rgb_img)
                plt.axis('off')

                plt.subplot(122)
                plt.title("Predicted")
                plt.imshow(drawn_mask_output)
                plt.axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("___")
            else:
                img_with_detections, detections = inference_output
                st.image(img_with_detections, width=display_width,
                         caption=f'Detection result for: {filename}')

        if image_type == "From MQTT":
            st.button("Refresh page")

            if st.checkbox("Run continuously", key='cbox_keep_running'):
                logger.info("Waiting for new image from MQTT ...")
                mqtt_msg_place = st.empty()
                while True:
                    mqtt_msg_place.info("Waiting for new image from MQTT ...")
                    sleep(1)
                    if session_state.refresh:
                        # logger.debug("REFRESHINGGGG")
                        # reset the state and refresh the page once received
                        session_state.refresh = False
                        st.experimental_rerun()

        st.stop()

    elif deploy_conf.input_type == 'Video':
        def update_conf_and_reset_video_deploy(conf_attr: str):
            update_deploy_conf(conf_attr)
            reset_video_deployment()

        # Does not seem to work properly
        # max_allowed_fps = st.sidebar.slider(
        #     'Maximum frame rate', 1, 60, 24, 1,
        #     key='selected_max_allowed_fps', on_change=reset_video_deployment,
        #     help="""This is the maximum allowed frame rate that the
        #         videostream will run at.""")
        if has_access:
            options = ("Uploaded Video", "Video Camera", "From MQTT")
            idx = options.index(deploy_conf.video_type)
            st.sidebar.radio(
                "Select type of video input",
                options, index=idx,
                key='video_type', help="MQTT should send image frames in bytes.",
                on_change=update_conf_and_reset_video_deploy, args=('video_type',))

            st.sidebar.slider(
                'Width of video (for display only)', 320, 1920,
                deploy_conf.video_width, 10,
                key='video_width',
                help="This is the width of video for visualization purpose.",
                on_change=update_deploy_conf, args=('video_width',)
            )

            # st.sidebar.checkbox(
            #     'Use video camera', value=deploy_conf.use_camera,
            #     key='use_camera',
            #     on_change=update_conf_and_reset_video_deploy, args=('use_camera',))
        else:
            st.sidebar.markdown(
                f"Video input type: **{deploy_conf.video_type}**")
            # if deploy_conf.use_camera:
            #     st.sidebar.markdown("Using **video camera** for deployment.")
            # else:
            #     st.sidebar.markdown("Using **uploaded video**.")
            st.sidebar.markdown(
                f"**Width of video**: {deploy_conf.video_width}")

        if deploy_conf.video_type == 'Video Camera':
            # TODO: test using streamlit-webrtc

            with st.sidebar.container():
                if has_access:
                    options = ('USB Camera', 'IP Camera')
                    idx = options.index(deploy_conf.camera_type)
                    st.radio(
                        "Select type of camera", options, index=idx,
                        key='camera_type', on_change=update_conf_and_reset_video_deploy,
                        args=('camera_type',))
                if deploy_conf.camera_type == 'USB Camera':
                    if has_access:
                        def reset_video_deployment_and_ports():
                            reset_video_deployment()
                            if 'working_ports' in session_state:
                                del session_state['working_ports']
                        st.button("Refresh camera ports", key='btn_refresh',
                                  on_click=reset_video_deployment_and_ports)

                        if not session_state.working_ports:
                            with st.spinner("Checking available camera ports ..."):
                                _, working_ports = list_available_cameras()
                                session_state.working_ports = working_ports.copy()
                            # stop if no camera port found
                            if not working_ports:
                                st.error(
                                    "No working camera source/port found.")
                                logger.error("No working camera port found")
                                st.stop()

                        st.radio(
                            "Select a camera port",
                            options=session_state.working_ports,
                            index=deploy_conf.camera_port,
                            key='camera_port',
                            on_change=update_conf_and_reset_video_deploy,
                            args=('camera_port',))
                    else:
                        st.markdown("USB Camera from camera port: "
                                    f"**{deploy_conf.camera_port}**")
                    video_source = deploy_conf.camera_port
                else:
                    if has_access:
                        st.text_input(
                            "Enter the IP address", value=deploy_conf.ip_cam_address,
                            key='ip_cam_address',
                            on_change=update_deploy_conf, args=(
                                'ip_cam_address',),
                            help="""This address could start with *http* or *rtsp*.
                            Most of the IP cameras  \nhave a username and password to access
                            the video. In such case,  \nthe credentials have to be provided
                            in the streaming URL as follow:  \n
                            **rtsp://username:password@192.168.1.64/1**""")
                    else:
                        st.markdown("IP Camera with address: "
                                    f"**{deploy_conf.ip_cam_address}** ")
                    video_source = deploy_conf.ip_cam_address

            # **************************** CSV FILE STUFF ****************************
            if 'today' not in session_state:
                # using this to keep track of the current day for updating CSV file,
                # store in session_state to take into account the case when user
                # decided to move to another page during deployment
                session_state.today = datetime.now().date()

            def update_retention_period():
                retention_period = session_state.day_input \
                    + (7 * session_state.week_input) \
                    + (30 * session_state.month_input)
                if retention_period == 0:
                    warning_place.warning(
                        "Retention period must be larger than 1 day!")
                    return
                # in 'days' unit
                deploy_conf.retention_period = retention_period

            # show CSV directory
            csv_path = deployment.get_csv_path(datetime.now())
            csv_dir = csv_path.parents[1]
            with st.sidebar.container():
                st.markdown("___")
                st.subheader("Info about saving results")
                st.markdown("#### Data retention period")
                warning_place = st.empty()
                if has_access:
                    with st.form('retention_period_form', clear_on_submit=True):
                        st.number_input("Day", 0, 1000, deploy_conf.retention_period, 1,
                                        key='day_input')
                        st.number_input("Week", 0, 10, 0, 1, key='week_input',
                                        help='7 days per week')
                        st.number_input("Month", 0, 12, 0, 1, key='month_input',
                                        help='30 days per month')
                        st.form_submit_button(
                            'Change retention period', on_click=update_retention_period)
                retention_period = deploy_conf.retention_period

                st.markdown(f"Retention period = **{retention_period} days**")
                with st.expander("CSV save file info"):
                    st.markdown(
                        f"**Inference results will be saved continuously in**: *{csv_dir}*  \n"
                        f"A new file will be created daily and files older than the "
                        f"retention period (`{retention_period} days`) will be deleted. "
                        "Be sure to click the `Pause deployment` button "
                        "to ensure the latest CSV file is saved properly if you have any "
                        "problem with opening the file.")
        elif deploy_conf.video_type == 'Uploaded Video':
            def temp_reset_vid():
                logger.debug("RESETTING VIDEO DEPLOY FOR VIDEO FILE")
                reset_video_deployment()
            video_file = st.sidebar.file_uploader(
                "Or upload a video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
                key='video_file_uploader', on_change=temp_reset_vid)
        else:
            logger.info("Using continuous frames receiving through MQTT")

        # **************************** MQTT STUFF ****************************
        saved_frame_dir = deployment.get_frame_save_dir('image')
        ng_frame_dir = deployment.get_frame_save_dir('NG')
        recording_dir = deployment.get_frame_save_dir('video')
        for save_dir in (saved_frame_dir, ng_frame_dir, recording_dir):
            if not save_dir.exists():
                os.makedirs(save_dir)

        if 'csv_writer' not in session_state:
            session_state.csv_writer = None
            session_state.csv_file = None

        if 'record' not in session_state:
            session_state.record = False
            session_state.vid_writer = None

        if 'check_labels' not in session_state:
            # for checking labels received through MQTT
            session_state.check_labels = None

        def start_publish_cb(client, userdata, msg):
            logger.info("Start publishing")
            session_state.publishing = True
            deploy_conf.publishing = True
            session_state.refresh = True

        def stop_publish_cb(client, userdata, msg):
            logger.info("Stopping publishing ...")
            # session_state.client_connected = False
            session_state.publishing = False
            deploy_conf.publishing = False
            deploy_conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True
            # st.success("Stopped publishing")
            # cannot use this within mqtt callback
            # st.experimental_rerun()

        def start_publish_frame_cb(client, userdata, msg):
            logger.info("Start publishing frames")
            deploy_conf.publish_frame = True
            session_state.refresh = True

        def stop_publish_frame_cb(client, userdata, msg):
            logger.info("Stopping publishing frames...")
            deploy_conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True

        def save_frame_cb(client, userdata, msg):
            logger.debug(f'Payload received for topic "{msg.topic}"')
            # need this to access to the frame from within mqtt callback
            nonlocal output_img, channels
            save_image(output_img, saved_frame_dir,
                       channels, deploy_conf.timezone)

        def start_record_cb(client, userdata, msg):
            session_state.record = True
            session_state.refresh = True

        def stop_record_cb(client, userdata, msg):
            session_state.record = False
            session_state.refresh = True

        def dobot_view_cb(client, userdata, msg):
            # view: str = msg.payload.decode()
            # logger.info(f"Received message from topic '{msg.topic}': "
            #             f"'{view}'")
            # session_state.check_labels = view
            payload = msg.payload
            res = deployment.validate_received_label_msg(payload)
            if res:
                logger.info("Received MQTT message with proper format from topic "
                            f"'{msg.topic}' to perform label checking: '{payload}'")
                session_state.check_labels = res
            else:
                session_state.check_labels = None

        topic_2cb = {
            topics.start_publish: start_publish_cb,
            topics.stop_publish: stop_publish_cb,
            topics.start_publish_frame: start_publish_frame_cb,
            topics.stop_publish_frame: stop_publish_frame_cb,
            topics.save_frame: save_frame_cb,
            topics.start_record: start_record_cb,
            topics.stop_record: stop_record_cb,
            topics.dobot_view: dobot_view_cb,
        }

        def add_video_callbacks():
            # on_message() will serve as fallback when none matched
            # or use this to be more precise on the subscription topic filter
            for topic, cb in topic_2cb.items():
                client.message_callback_add(topic, cb)

        st.sidebar.markdown("___")
        st.sidebar.subheader("MQTT Options")

        # set up callbacks for video deployment
        if not session_state.added_video_cbs:
            logger.debug(
                "Adding callbacks to MQTT client for video deployment")
            add_video_callbacks()
            session_state.added_video_cbs = True

        # NOTE: Docker needs to use service name instead to connect to broker,
        # but user should always connect to 'localhost' or this PC's IP Address
        st.sidebar.info(f"**MQTT broker**: localhost  \n**Port**: {conf.port}")

        if has_access:
            topic_error_place = st.sidebar.empty()

            # def update_conf_topic(topic_attr: str):
            def update_conf_topic():
                for topic_attr in topics.__dict__.keys():
                    new_topic = session_state[topic_attr]
                    if new_topic == '':
                        logger.error('Topic cannot be empty string')
                        topic_error_place.error('Topic cannot be empty string')
                        sleep(1)
                        st.experimental_rerun()

                    previous_topic = getattr(topics, topic_attr)

                    if new_topic == previous_topic:
                        # no need to change anything if user didn't change the topic
                        continue

                    # unsubscribe the old topic and remove old topic callback
                    client.unsubscribe(previous_topic)
                    client.message_callback_remove(previous_topic)

                    # update MQTTTopics with new topic, add callback, and subscribe
                    setattr(topics, topic_attr, new_topic)

                    callback_func = topic_2cb.get(previous_topic)
                    if callback_func:
                        # only add callbacks for the topics that have callbacks
                        client.message_callback_add(new_topic, callback_func)
                    client.subscribe(new_topic, qos=conf.qos)

                    logger.info(f"Updated MQTTTopics.{topic_attr} from {previous_topic} "
                                f"to {new_topic}")

            st.sidebar.radio(
                'MQTT QoS', (0, 1, 2), conf.qos, key='mqtt_qos',
                on_change=update_mqtt_qos)

            st.sidebar.markdown("**MQTT Topics**")
            st.sidebar.markdown(
                "If you change any MQTT topic name(s), please click the **Update Config** "
                "button to allow the changes to be made.")
            # must clear on submit to show the correct values on form
            # NOTE: the key name must be the same as the topic attribute name
            with st.sidebar.form('form_mqtt_topics', clear_on_submit=True):
                st.text_input(
                    'Publishing output frames to', topics.publish_frame,
                    key='publish_frame', help="This is used to publish output frames. "
                    "Our MQTT client is not subscribed to this topic.")
                st.text_input(
                    'Publishing results to', topics.publish_results,
                    key='publish_results', help="This is used to publish inference results. "
                    "Our MQTT client is not subscribed to this topic.")

                st.text_input(
                    'Publish frame to our app', topics.recv_frame,
                    key='recv_frame', help="Publish the image frame in bytes to this topic "
                    "for MQTT input deployment")
                st.text_input(
                    'Start publishing results', topics.start_publish,
                    key='start_publish')
                st.text_input(
                    'Stop publishing results', topics.stop_publish,
                    key='stop_publish')
                st.text_input(
                    'Start publishing frames', topics.start_publish_frame,
                    key='start_publish_frame')
                st.text_input(
                    'Stop publishing frames', topics.stop_publish_frame,
                    key='stop_publish_frame')
                st.text_input(
                    'Save current frame', topics.save_frame,
                    key='save_frame')
                st.text_input(
                    'Start recording frames', topics.start_record,
                    key='start_record')
                st.text_input(
                    'Stop recording frames', topics.stop_record,
                    key='stop_record')
                st.text_input(
                    'Current camera view and required labels to check',
                    topics.dobot_view, key='dobot_view',
                    help='''Send a JSON object containing the "view" and "labels" to check
                    whether the detected labels at the current view contains the "labels"
                    sent from MQTT. The value for **"view"** must only be a string, while
                    the value for the **"labels"** must be a list. Example JSON object: 
                    `"{"view": "top", "labels": ["date", "white dot"]}"''')
                st.form_submit_button(
                    "Update Config", on_click=update_conf_topic,
                    help="Please press this button to update if you change any MQTT "
                    "topic name(s).")
        else:
            st.sidebar.markdown(
                f"MQTT QoS is set to level **{conf.qos}**")
            st.sidebar.info(
                "#### Publishing Results to MQTT Topic:  \n"
                f"{topics.publish_results}  \n"
                "#### Publishing output frames to:  \n"
                f"{topics.publish_frame}"
            )
            st.sidebar.info(
                "#### Subscribed MQTT Topics:  \n"
                f"**Receiving input frames**: {topics.recv_frame}  \n"
                f"**Start publishing results**: {topics.start_publish}  \n"
                f"**Stop publishing results**: {topics.stop_publish}  \n"
                f"**Start publishing frames**: {topics.start_publish_frame}  \n"
                f"**Stop publishing frames**: {topics.stop_publish_frame}  \n"
                f"**Save current frame**: {topics.save_frame}  \n"
                f"**Start recording frames**: {topics.start_record}  \n"
                f"**Stop recording frames**: {topics.stop_record}")
        with st.sidebar.expander("Notes"):
            st.markdown(
                f"Make sure to connect to **'localhost'** broker (or IP Address of this PC) "
                f"with the correct port **{conf.port}**. "
                "Then just publish an arbitrary message to any of the subscribed MQTT "
                "topics to trigger the functionality (except for the first two topics, "
                "where our MQTT client is not subsrcibed to).  \nFor the **saved "
                f"frames**, they will be saved in your project's folder at *{saved_frame_dir}*, "
                f"while recorded video will be saved at *{recording_dir}*. Please "
                "**do not simply delete these folders during deployment**, "
                "otherwise error will occur. You can delete them after pausing/ending "
                "the deployment if you wish.  \n"
                "For the **publishing frames**, they are in **bytes** format.")

        # ************************ Video deployment button ************************

        # allow the user to click the "Deploy Model button" after done configuring everything
        if deploy_conf.input_type == 'Video':
            if deploy_conf.video_type == 'Video Camera':
                # only show these if a camera is not selected and not deployed yet
                if not session_state.deployed:
                    if not deploy_btn_place.button(
                        "🛠️ Deploy Model", key='btn_deploy_cam',
                            help='Deploy your model with the selected camera source'):
                        st.stop()

                    with st.spinner("Loading up camera ..."):
                        # NOTE: VideoStream does not work with filepath
                        try:
                            session_state.camera = WebcamVideoStream(
                                src=video_source).start()
                            if session_state.camera.read() is None:
                                raise Exception(
                                    "Video source is not valid")
                        except Exception as e:
                            st.error(
                                f"Unable to read from video source {video_source}")
                            logger.error(
                                f"Unable to read from video source {video_source}: {e}")
                            st.stop()

                        session_state.deployed = True
                        sleep(2)  # give the camera some time to sink in
                        # rerun just to avoid displaying unnecessary buttons
                        st.experimental_rerun()
            elif deploy_conf.video_type == 'Uploaded Video':
                if not video_file:
                    st.stop()

                # use this to avoid keep calling the file_uploader's callback
                # NOTE: still not helping...
                uploaded_video = deepcopy(video_file)

                video_path = str(TEMP_DIR / uploaded_video.name)

                if not session_state.deployed:
                    if not deploy_btn_place.button(
                        "🛠️ Deploy Model", key='btn_deploy_uploaded_vid',
                            help='Deploy your model with the uploaded video'):
                        st.stop()

                    if TEMP_DIR.exists():
                        shutil.rmtree(TEMP_DIR)
                    os.makedirs(TEMP_DIR)
                    logger.debug(f"{video_path = }")
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_video.getvalue())

                    try:
                        session_state.camera = cv2.VideoCapture(video_path)
                        assert session_state.camera.isOpened(), "Video is unreadable"
                    except Exception as e:
                        st.error(f"Unable to read from the video file: "
                                 f"'{uploaded_video.name}'")
                        logger.error(f"Unable to read from the video file: "
                                     f"'{uploaded_video.name}' with error: {e}")
                        st.stop()

                    session_state.deployed = True
                    sleep(2)  # give the camera some time to sink in
                    st.experimental_rerun()
            else:
                if not session_state.deployed:
                    if not deploy_btn_place.button(
                            "🛠️ Deploy Model", key='btn_deploy_vid_mqtt',
                            help='Deploy your model for frames sent from MQTT'):
                        st.stop()

                    session_state.deployed = True
                logger.info(
                    "Deploying for video frames received from MQTT")

        # after user has clicked the "Deploy Model" button
        if session_state.deployed:
            deploy_btn_place.button(
                "Pause Deployment", key='btn_pause_deploy',
                on_click=reset_video_deployment,
                help=("Pause deployment after you have deployed the model  \n"
                      "with a running video camera. Or use this to reset  \n"
                      "camera if there is any problem with loading up  \n"
                      "the camera. Note that this is extremely important  \n"
                      "to ensure your camera is properly stopped and the  \n"
                      "camera access is given back to your system. This will  \n"
                      "also save the latest CSV file in order to be opened."))

            if deploy_conf.video_type != 'From MQTT':
                if deploy_conf.video_type == 'Video Camera':
                    stream = session_state.camera.stream
                    deploy_status_place.info(
                        "**Status**: Deployed for video camera input")
                else:
                    stream = session_state.camera
                    deploy_status_place.info(
                        "**Status**: Deployed for uploaded video")

                    with deploy_status_col:
                        st.subheader('Input Video')
                        try:
                            st.video(video_path)
                        except Exception as e:
                            st.error(f"Unable to read from the video file: "
                                     f"'{uploaded_video.name}'")
                            logger.error(f"Unable to read from the video file: "
                                         f"'{uploaded_video.name}' with error: {e}")
                            st.stop()

                width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_input = int(stream.get(cv2.CAP_PROP_FPS))
                logger.info(
                    f"Video properties: {width = }, {height = }, {fps_input = }")
            else:
                msg_place = st.empty()
                if not session_state.mqtt_recv_frame:
                    logger.info("Waiting for frames from MQTT ...")
                    while not session_state.mqtt_recv_frame:
                        msg_place.info(
                            "No frame received from MQTT. Waiting ...")
                        sleep(1)
                msg_place.info("Frame received **from MQTT**")
                logger.info("Reading the frame from MQTT")

                frame = image_from_buffer(session_state.mqtt_recv_frame)
                height, width = frame.shape[:2]
                logger.info("Properties of received frame: "
                            f"{width = }, {height = }")
        else:
            # stop if the "Deploy Model" button is not clicked yet
            logger.info("Model is not deployed yet")
            st.stop()

        # *********************** DOBOT arm demo ***********************
        # DOBOT_TASK = dobot_demo.DobotTask.Box  # for box shapes
        # DOBOT_TASK = dobot_demo.DobotTask.P2_143  # for machine part
        DOBOT_TASK = dobot_demo.DobotTask.DEBUG  # for debugging publishing MQTT
        run_func = dobot_demo.run

        if DOBOT_TASK == dobot_demo.DobotTask.Box:
            VIEW_LABELS = dobot_demo.BOX_VIEW_LABELS
        elif DOBOT_TASK == dobot_demo.DobotTask.P2_143:
            VIEW_LABELS = dobot_demo.P2_143_VIEW_LABELS
        elif DOBOT_TASK == dobot_demo.DobotTask.DEBUG:
            VIEW_LABELS = dobot_demo.DEBUG_VIEW_LABELS
            run_func = dobot_demo.debug_run

        st.button("Move DOBOT and detect",
                  key='btn_move_dobot', on_click=run_func,
                  args=(conf, DOBOT_TASK))

        # *********************** Deployment video loop ***********************
        def create_video_writer_if_not_exists():
            if not session_state.vid_writer:
                logger.info("Creating video file to record to")
                # NOTE: THIS VIDEO SAVE FORMAT IS VERY PLATFORM DEPENDENT
                # TODO: THE VIDEO FILE MIGHT NOT SAVE PROPERLY
                # usually either MJPG + .avi, or XVID + .mp4
                FOURCC = cv2.VideoWriter_fourcc(*"XVID")
                filename = f"video_{get_now_string(timezone=deploy_conf.timezone)}.mp4"
                video_save_path = str(recording_dir / filename)
                # st.info(f"Video is being saved to **{video_save_path}**")
                logger.info(f"Video is being saved to '{video_save_path}'")
                # this FPS value is the FPS of the output video file,
                # note that if this value is much higher than the fps
                # during the inference time, the output video will look
                # like it's moving very very fast
                FPS = 24
                session_state.vid_writer = cv2.VideoWriter(
                    video_save_path, FOURCC, FPS,
                    (width, height), True)

        st.subheader("Output Video")
        show_video_col = st.container()
        msg_cont, _ = st.columns(2)
        with msg_cont:
            msg_place = st.empty()
        output_video_place = st.empty()
        publish_place = st.sidebar.empty()
        fps_col, width_col, height_col = st.columns(3)

        with show_video_col:
            show_video = st.checkbox(
                'Show video', value=True, key='show_video')
            draw_result = st.checkbox(
                "Draw labels", value=True, key='draw_result')

            def update_record(is_recording: bool):
                session_state.record = is_recording

            if not session_state.record:
                st.button('Start recording', key='btn_start_record',
                          help=f"The video will be saved in *{recording_dir}*",
                          on_click=update_record, args=(True,))
            else:
                st.button("Stop recording and save the video",
                          key='btn_stop_and_save_vid',
                          on_click=update_record, args=(False,))

        if has_access:
            def update_publishing_conf(is_publishing: bool):
                deploy_conf.publishing = is_publishing
                session_state.publishing = is_publishing
                # also change whether publishing frame or not
                deploy_conf.publish_frame = is_publishing

            def update_publish_frame_conf(publish_frame: bool):
                deploy_conf.publish_frame = publish_frame

            if session_state.publishing:
                # using buttons to allow the widget to change after rerun
                # whereas checkbox does not change after rerun
                publish_place.button("Stop publishing results", key='btn_stop_pub',
                                     on_click=update_publishing_conf, args=(False,))
            else:
                publish_place.button("Start publishing results", key='btn_start_pub',
                                     on_click=update_publishing_conf, args=(True,))

            if deploy_conf.publish_frame:
                st.sidebar.button(
                    "Stop publishing frames", key='btn_stop_pub_frame',
                    help="Stop publishing frames as bytes to the MQTT Topic: "
                    f"*{topics.publish_frame}*.",
                    on_click=update_publish_frame_conf, args=(False,))
            else:
                st.sidebar.button(
                    "Start publishing frames", key='btn_start_pub_frame',
                    help="Publish frames as bytes to the MQTT Topic:  \n"
                    f"*{topics.publish_frame}*.  \nNote that this could significantly "
                    "reduce FPS.", on_click=update_publish_frame_conf, args=(True,))
        else:
            session_state.publishing = deploy_conf.publishing
            if session_state.publishing:
                st.markdown("Currently is publishing results to the topic: "
                            f"*{topics.publish_results}*")
            else:
                st.markdown(
                    "Currently is not publishing any results through MQTT.")

            if deploy_conf.publish_frame:
                st.markdown("Currently is publishing output frames to the topic: "
                            f"*{topics.publish_frame}*")
            else:
                st.markdown(
                    "Currently is not publishing any output frames through MQTT.")

        show_labels = st.checkbox("Show the detected labels", value=True)
        if show_labels:
            result_col = st.container()
            with result_col:
                st.markdown("**Detected Results**")
                result_place = st.markdown("Coming up")
        with fps_col:
            st.markdown("**Frame Rate**")
            fps_place = st.markdown("0")
        with width_col:
            st.markdown("**Frame Width**")
            st.markdown(kpi_format(width), unsafe_allow_html=True)
        with height_col:
            st.markdown("**Frame Height**")
            st.markdown(kpi_format(height), unsafe_allow_html=True)
        st.markdown("___")

        # prepare variables for the video deployment loop
        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=draw_result, **pipeline_kwargs)

        if DEPLOYMENT_TYPE == 'Image Classification':
            is_image_classif = True
            get_result_fn = partial(deployment.get_classification_results,
                                    timezone=deploy_conf.timezone)
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            is_image_classif = False
            get_result_fn = partial(deployment.get_segmentation_results,
                                    timezone=deploy_conf.timezone)
        else:
            is_image_classif = False
            get_result_fn = partial(deployment.get_detection_results,
                                    conf_threshold=deploy_conf.confidence_threshold,
                                    timezone=deploy_conf.timezone)
        publish_func = partial(client.publish,
                               topics.publish_results, qos=conf.qos)
        publish_frame_fn = partial(client.publish,
                                   topics.publish_frame, qos=conf.qos)

        starting_time = datetime.now()
        csv_path = deployment.get_csv_path(starting_time)
        csv_dir = csv_path.parent
        if not csv_dir.exists():
            os.makedirs(csv_dir)
        logger.info(f'Operation begins at: {starting_time.isoformat()}')
        logger.info(f'Inference results will be saved in {csv_dir}')
        # use_cam = deploy_conf.use_camera
        if deploy_conf.video_type == 'Video Camera':
            video_type = 0
        elif deploy_conf.video_type == 'From MQTT':
            video_type = 1
        else:
            video_type = 2
        display_width = deploy_conf.video_width
        publish_frame = deploy_conf.publish_frame
        fps = 0
        prev_time = 0
        first_csv_save = True

        # start the video deployment loop
        while True:
            if session_state.refresh:
                # refresh page once to refresh the widgets
                logger.debug("REFRESHING PAGEE")
                session_state.refresh = False
                st.experimental_rerun()

            if video_type == 0:
                frame = session_state.camera.read()
            elif video_type == 1:
                frame = image_from_buffer(session_state.mqtt_recv_frame)
            else:
                ret, frame = session_state.camera.read()
                if not ret:
                    break

            # frame.flags.writeable = True  # might need this?
            # run inference on the frame
            inference_output = inference_pipeline(frame)
            if is_image_classif:
                results = get_result_fn(*inference_output)
                output_img = frame
                # the read frame is in BGR format
                channels = 'BGR'
            else:
                if draw_result:
                    output_img, pred = inference_output
                    channels = 'RGB'
                else:
                    pred = inference_output
                    output_img = frame
                    # the read frame is in BGR format
                    channels = 'BGR'
                results = get_result_fn(pred)

            if show_video:
                output_video_place.image(output_img, channels=channels,
                                         width=display_width)

            if session_state.record:
                # need to be within the video loop to ensure we also get the latest
                #  session_state updates from MQTT callback
                msg_place.info("Recording ...")
                with show_video_col:
                    create_video_writer_if_not_exists()
                if channels == 'RGB':
                    # cv2.VideoWriter needs BGR format
                    out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                else:
                    out = output_img
                session_state.vid_writer.write(out)
            else:
                if session_state.vid_writer:
                    msg_place.empty()
                    logger.info("Saving recorded file")
                    # must release to close the video file
                    session_state.vid_writer.release()
                    session_state.vid_writer = None

            # count FPS
            curr_time = perf_counter()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            fps_place.markdown(kpi_format(int(fps)), unsafe_allow_html=True)

            if show_labels:
                result_place.table(results)

            if publish_frame:
                frame_bytes = image_to_bytes(output_img, channels)
                info = publish_frame_fn(frame_bytes)

            # NOTE: this session_state is currently ONLY used for DOBOT arm for
            # object detection demo to detect different labels at different views
            if session_state.check_labels:
                # session_state.check_labels should be Tuple[str, List[str]]
                # generated from validate_received_label_msg() in dobot_view_cb()
                view, required_labels = session_state.check_labels

                if view == 'end':
                    logger.info("Ending the label checking process")
                    # clear the message if the robot motion is ended
                    msg_place.empty()
                    # and reset back to None
                    session_state.check_labels = None
                    continue

                # required_label_cnts = VIEW_LABELS[view]
                # if dobot_demo.check_result_labels(results, required_label_cnts):
                if deployment.check_labels(results, required_labels, DEPLOYMENT_TYPE):
                    logger.info(f"All labels present at '{view}' view")
                    msg_place.success(f"### {view.upper()} view: OK")
                else:
                    logger.warning("Required labels are not detected at "
                                   f"'{view}' view")
                    msg_place.error(f"### {view.upper()} view: NG")
                    save_image(output_img, ng_frame_dir,
                               channels, timezone=deploy_conf.timezone, prefix=view)
                    logger.info(f"NG image saved at {ng_frame_dir}")

                # set this to None to ONLY CHECK FOR ONCE for the same view
                session_state.check_labels = None

            if not results:
                continue

            if session_state.publishing:
                payload = json.dumps(results)
                info = publish_func(payload=payload)

            # save results to CSV file only if using video camera
            if video_type != 0:
                continue

            if first_csv_save:
                first_csv_save = False
                if not csv_path.exists():
                    new_file = True
                else:
                    new_file = False
                create_csv_file_and_writer(
                    csv_path, results, new_file=new_file)
            now = datetime.now()
            today = now.date()
            if today > session_state.today:
                session_state.csv_file.close()

                deployment.delete_old_csv_files(
                    retention_period)

                csv_path = deployment.get_csv_path(now)
                session_state.today = today
                create_csv_file_and_writer(csv_path, results)
            else:
                for row in results:
                    session_state.csv_writer.writerow(row)

            # This below does not seem to work properly
            # if fps > max_allowed_fps:
            #     sleep(1 / (fps - max_allowed_fps))

        # clean up everything if it's an uploaded video
        if session_state.vid_writer:
            logger.info("Saving recorded file")
            # must release to close the video file
            session_state.vid_writer.release()
            session_state.vid_writer = None

        reset_video_deployment()

        if TEMP_DIR.exists():
            logger.debug("Removing temporary directory")
            shutil.rmtree(TEMP_DIR)

        st.info("Inference done for uploaded video.")


def main():
    # False for debugging
    index(RELEASE=False)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
