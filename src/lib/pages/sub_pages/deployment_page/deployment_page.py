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

from datetime import datetime
from functools import partial
import json
import os
import shutil
import sys
from time import perf_counter, sleep
from itertools import cycle
import gc
from typing import Any, Callable, Dict, List

import cv2
# from imutils.video.webcamvideostream import WebcamVideoStream
from core.webcam.webcamvideostream import CameraFailError, WebcamVideoStream
from matplotlib import pyplot as plt
import numpy as np
from paho.mqtt.client import Client
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state
# this is changed from add_report_ctx as of Streamlit v1.4.0
from streamlit.script_run_context import add_script_run_ctx
import tensorflow as tf

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
from core.utils.helper import (Timer, get_all_timezones, get_now_string,
                               list_available_cameras, save_image, get_ram_usage,
                               reset_camera)
from user.user_management import User, UserRole
from data_manager.database_manager import init_connection
from project.project_management import Project
from data_manager.dataset_management import Dataset
from machine_learning.visuals import create_color_legend
from deployment.deployment_management import DeploymentConfig, DeploymentPagination, Deployment
from deployment.utils import (ORI_PUBLISH_FRAME_TOPIC, MQTTConfig, MQTTTopics,
                              create_csv_file_and_writer, image_from_buffer, image_to_bytes,
                              get_mqtt_client, read_images_from_uploaded, reset_video_deployment)
from dobot_arm_demo import main as dobot_demo
from csv_label_inspection import csv_label_check as csv_labels
from Node_Red.Not_dobot_version import Label_View
# from Node_Red.Dobot_Version import Dobot_Version

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
        session_state.working_ports = []
    if 'deployment_conf' not in session_state:
        # to store the config in case the user needs to go to another page during deployment
        session_state.deployment_conf = DeploymentConfig()
    if 'client' not in session_state:
        session_state.client = get_mqtt_client()
        session_state.client_connected = False
        # to check whether video callbacks have been added to MQTT client
        session_state.added_video_cbs = False
        session_state.mqtt_conf = MQTTConfig()

        # to check whether is publishing results through MQTT
        session_state.publishing = True
    if 'mqtt_recv_frame' not in session_state:
        # to check whether any image/frame is received through MQTT
        session_state.mqtt_recv_frame = None

    conf: DeploymentConfig = session_state.deployment_conf
    client: Client = session_state.client
    mqtt_conf: MQTTConfig = session_state.mqtt_conf
    topics: MQTTTopics = session_state.mqtt_conf.topics

    project: Project = session_state.project
    deployment: Deployment = session_state.deployment
    user: User = session_state.user
    DEPLOYMENT_TYPE = deployment.deployment_type

    def reset_image_idx():
        """Reset index of image chosen to avoid exceeding the max number of images
        on widget changes"""
        session_state.image_idx = 0

    def reset_single_camera_conf():
        conf.use_multi_cam = False
        conf.num_cameras = 1
        # reset camera titles
        conf.camera_titles = ['']
        topics.publish_frame = [topics.publish_frame[0]]

    def update_camera_sources_from_types():
        """Update camera sources according to selected camera types"""
        available_ports = iter(session_state.working_ports)
        conf.camera_sources = []
        for cam_type in conf.camera_types:
            if cam_type == 'USB Camera':
                try:
                    curr_source = next(available_ports)
                except StopIteration:
                    # use empty string just in case there is no working_ports yet
                    curr_source = ''
            else:
                # empty string for IP camera
                curr_source = ''
            conf.camera_sources.append(curr_source)

    def reset_multi_camera_config():
        """Reset camera titles and camera sources according to number
        of cameras selected
        """
        conf.camera_titles = []
        conf.camera_types = []
        conf.camera_sources = []
        for _ in range(conf.num_cameras):
            conf.camera_titles.append('')
            conf.camera_types.append('IP Camera')
            conf.camera_sources.append('')

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
            client.subscribe(topic, qos=mqtt_conf.qos)

    def update_mqtt_qos():
        # take from the widget's state and save to our mqtt_conf
        logger.info(f"Updated QoS level from {mqtt_conf.qos} to "
                    f"{session_state.mqtt_qos}")
        mqtt_conf.qos = int(session_state.mqtt_qos)

        subscribe_topics(resubscribe=True)

    def end_deployment():
        Deployment.reset_deployment_page()
        session_state.deployment_pagination = DeploymentPagination.Models

    def update_deploy_conf(conf_attr: str):
        """Update deployment config on any change of the widgets.

        NOTE: `conf_attr` must exist in the `session_state` (usually is the widget's state)
        and must be the same with the `DeploymentConfig` attribute's name."""
        val = session_state[conf_attr]
        logger.debug(f"New value for {conf_attr}: {val}")
        val_type = conf.__annotations__[conf_attr]
        # logger.debug(f"{val_type = }")
        # logger.debug(f"{type(val_type) = }")
        if val_type == "int":
            # cast to int to avoid issues when changing the widgets too fast
            logger.debug(f"Casting DeploymentConfig.{conf_attr} to int")
            val = int(val)
        logger.info(f"Updated deployment config: {conf_attr} = {val}")
        setattr(conf, conf_attr, val)

    def load_multi_webcams(stop_on_error: bool = True):
        conf.camera_keys = []
        for i, src in enumerate(conf.camera_sources):
            with st.spinner(f"Loading up camera {i} ..."):
                # NOTE: these keys must start with 'camera' to be able to
                # reset them in reset_camera()
                cam_type = conf.camera_types[i]
                if cam_type == 'USB Camera':
                    is_usb_camera = True
                else:
                    # set this for IP camera to avoid release it
                    # during reset_camera() to avoid error
                    is_usb_camera = False

                cam_key = f'camera_{i}'
                conf.camera_keys.append(cam_key)

                try:
                    session_state[cam_key] = WebcamVideoStream(
                        src=src, is_usb_camera=is_usb_camera, name=cam_key).start()
                    if session_state[cam_key].read() is None:
                        raise CameraFailError(
                            "Video source is not valid")
                except CameraFailError as e:
                    if src == '':
                        error_msg_place.error(
                            f"Unable to read from empty IP camera address {i}.")
                    else:
                        error_msg_place.error(
                            f"Unable to read from {cam_type} {i}: '{src}'. Most likely "
                            "because the camera port is incorrect and you need to try "
                            "to refresh it. Or maybe the camera is disconnected or "
                            "being occupied by another process.")
                    logger.error(
                        f"Unable to read from {cam_type} {i} from source {src}: {e}")
                    reset_camera()
                    if stop_on_error:
                        st.stop()
                    else:
                        raise e

        session_state.deployed = True
        sleep(2)  # give the cameras some time to sink in
        # rerun just to avoid displaying unnecessary buttons
        # st.experimental_rerun()

    def update_conf_topic():
        """To compare current and previous topics to update the MQTT topics"""
        for topic_attr in topics.__dict__.keys():
            for i in range(conf.num_cameras):
                if topic_attr == 'publish_frame':
                    # only this attribute is a List[str],
                    # for image deployment, num_cameras already reset to 1
                    #   in reset_single_camera_conf()
                    state_key = f'publish_frame_{i}'
                    previous_topic = topics.publish_frame[i]
                else:
                    if i > 0:
                        # don't need to check the same topic_attr more than once
                        break
                    state_key = topic_attr
                    previous_topic = getattr(topics, topic_attr)

                new_topic = session_state[state_key]
                if new_topic == '':
                    logger.error('Topic cannot be empty string')
                    error_msg_place.error(
                        'Topic cannot be empty string')
                    sleep(1)
                    st.experimental_rerun()

                if new_topic == previous_topic:
                    # no need to change anything if user didn't change the topic
                    continue

                # unsubscribe the old topic and remove old topic callback
                client.unsubscribe(previous_topic)
                client.message_callback_remove(previous_topic)

                # update MQTTTopics with new topic, add callback, and subscribe
                if topic_attr == 'publish_frame':
                    topics.publish_frame[i] = new_topic
                else:
                    setattr(topics, topic_attr, new_topic)

                callback_func = topic_2cb.get(previous_topic)
                if callback_func is not None:
                    # only add callbacks for the topics that have callbacks
                    client.message_callback_add(
                        new_topic, callback_func)
                client.subscribe(new_topic, qos=mqtt_conf.qos)

                logger.info(f"Updated MQTTTopics.{topic_attr} from {previous_topic} "
                            f"to {new_topic}")

    def get_result_postprocessor() -> Callable[..., List[Dict[str, Any]]]:
        if DEPLOYMENT_TYPE == 'Image Classification':
            get_result_fn = partial(deployment.get_classification_results,
                                    timezone=conf.timezone)
        elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            get_result_fn = partial(deployment.get_segmentation_results,
                                    timezone=conf.timezone)
        else:
            get_result_fn = partial(deployment.get_detection_results,
                                    conf_threshold=conf.confidence_threshold,
                                    timezone=conf.timezone)
        return get_result_fn

    # connect MQTT broker and set up callbacks
    if not session_state.client_connected:
        logger.debug(f"{mqtt_conf = }")
        with st.spinner("Connecting to MQTT broker ..."):
            try:
                client.connect(mqtt_conf.broker, port=mqtt_conf.port)
            except Exception as e:
                st.error("Error connecting to MQTT broker. Please make sure MQTT "
                         "broker is installed and turned on in your system.")
                if os.getenv('DEBUG', '1') == '1':
                    st.exception(e)
                logger.error(
                    f"Error connecting to MQTT broker {mqtt_conf.broker}: {e}")
                st.stop()

            sleep(2)  # Wait for connection setup to complete
            logger.info("MQTT client connected successfully to "
                        f"{mqtt_conf.broker} on port {mqtt_conf.port}")

            subscribe_topics()

            # add callbacks for image input type first because the input type
            # defaults to "Image"
            client.message_callback_add(
                topics.recv_frame, image_recv_frame_cb)

            client.loop_start()

            # need to add this to avoid Missing ReportContext error
            # https://github.com/streamlit/streamlit/issues/1326
            # NOTE: it's changed from add_report_context() to this since Streamlit v1.4.0
            add_script_run_ctx(client._thread)

            session_state.client_connected = True

    st.subheader("Deployment Status")

    if not tf.config.list_physical_devices('GPU'):
        st.warning("""**WARNING**: You don't have access to GPU. Inference time will
        be slower depending on the capability of your CPU and system.""")

    deploy_status_place = st.empty()
    deploy_status_place.info("**Status**: Not deployed. Please upload an image/video "
                             "or use video camera or MQTT and click **\"Deploy Model\"**.")

    st.warning("**NOTE**: Please do not simply refresh the page without ending "
               "deployment or errors could occur.")

    col1, col2, _, _ = st.columns([1, 1, 1, 2.5])

    with col2:
        deploy_btn_place = st.empty()

    col1.button(
        "End Deployment", key='btn_stop_image_deploy',
        on_click=end_deployment,
        help="This will end the deployment and reset the entire  \n"
        "deployment configuration. Please **make sure** to use this  \n"
        "button to stop deployment before proceeding to any other  \n"
        "page! Or use the **pause button** (only available for camera  \n"
        "input) if you want to pause the deployment to do any other  \n"
        "things without resetting, such as switching user, or viewing  \n"
        "the latest saved CSV file (only for video camera deployment).")

    reset_camera_place = st.empty()
    st.markdown("___")

    error_msg_place = st.empty()

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
                # reset to default camera config state
                reset_single_camera_conf()
                client.message_callback_add(
                    topics.recv_frame, image_recv_frame_cb)

            conf.input_type = session_state.input_type

        all_timezones = get_all_timezones()
        tz_idx = all_timezones.index(conf.timezone)
        st.sidebar.selectbox(
            "Local Timezone", all_timezones, index=tz_idx, key='timezone',
            help="Select your local timezone to have the correct time output in results.",
            on_change=update_deploy_conf, args=('timezone',))

        options = ('Image', 'Video')
        idx = options.index(conf.input_type)
        st.sidebar.radio(
            'Choose the Type of Input', options,
            index=idx, key='input_type',
            on_change=update_input_type_conf)
    else:
        st.sidebar.markdown(f"**Selected timezone**: {conf.timezone}")
        st.sidebar.markdown(f"**Input type**: {conf.input_type}")

    options_col, _ = st.columns(2)

    if DEPLOYMENT_TYPE == 'Image Classification':
        pipeline_kwargs = {}
    elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
        ignore_background = st.sidebar.checkbox(
            "Ignore background", value=conf.ignore_background,
            key='ignore_background',
            help="Ignore background class for visualization purposes.  \n"
            "Note that turning this on could significantly reduce the FPS.",
            on_change=update_deploy_conf, args=('ignore_background',))
        legend = create_color_legend(
            deployment.class_colors, bgr2rgb=False,
            ignore_background=ignore_background)
        st.sidebar.markdown("**Legend**")
        st.sidebar.image(legend)
        pipeline_kwargs = {'ignore_background': ignore_background}
    else:
        if has_access:
            options_col.slider(
                "Confidence threshold:",
                min_value=0.1,
                max_value=0.99,
                value=conf.confidence_threshold,
                step=0.01,
                format='%.2f',
                key='confidence_threshold',
                help=("If a prediction's confidence score exceeds this threshold, "
                      "then it will be displayed, otherwise discarded."),
                on_change=update_deploy_conf, args=('confidence_threshold',)
            )
        else:
            options_col.markdown(
                f"**Confidence threshold**: {conf.confidence_threshold}")
        pipeline_kwargs = {'conf_threshold': conf.confidence_threshold}

    if conf.input_type == 'Image':
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
            display_width = st.sidebar.slider(
                "Select width of image to resize for display",
                35, 1000, 500, 5, key='display_width')

        st.sidebar.markdown("___")
        st.sidebar.subheader("MQTT QoS")
        st.sidebar.radio(
            'MQTT QoS', (0, 1, 2), mqtt_conf.qos, key='mqtt_qos',
            on_change=update_mqtt_qos)

        with st.sidebar.form('form_img_mqtt_topics', clear_on_submit=True):
            st.subheader("MQTT Topics")
            st.text_input(
                'Publish frame to our app', topics.recv_frame,
                key='image_recv_frame', help="Publish the image frame in bytes/buffer "
                "to this topic for MQTT input deployment.")
            st.text_input(
                'Publishing results to', topics.publish_results,
                key='publish_results', help="This is used to publish inference results "
                "to the outside. Our MQTT client is not subscribed to this topic.")
            st.text_input(
                f'Publishing frames for to', topics.publish_frame[0],
                key='publish_frame_0', help="This is used to publish output "
                "images. Our MQTT client is not subscribed to this topic.")
            st.form_submit_button(
                "Update Config", on_click=update_conf_topic,
                help="Please press this button to update if you change any MQTT "
                "topic name(s).")

        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=True, **pipeline_kwargs)
        get_result_fn = get_result_postprocessor()

        deploy_status_place.info("**Status**: Deployed for input images")

        if DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
            with st.expander("Notes about deployment for semantic segmentation"):
                st.markdown(
                    """If this is an externally trained segmentation model, it might 
                    not work properly for making predictions for our app's implementation.
                    Please try with sample images or uploaded images first to see the results. 
                    If there is anything wrong, please do not proceed to video deployment.""")

        for img, filename in imgs_info:
            st.subheader(filename)
            ori_image_height, ori_image_width = img.shape[:2]
            st.markdown(
                f"Original image resolution: **{ori_image_width} x {ori_image_height}**")

            with st.spinner("Running inference ..."):
                try:
                    with Timer("Inference on image"):
                        inference_output = inference_pipeline(img)
                except Exception as e:
                    if os.getenv('DEBUG', '1') == '1':
                        st.exception(e)
                    logger.error(
                        f"Error running inference with the model: {e}")
                    error_msg_place.error("""Error when trying to run inference with the model,
                        please check with Admin/Developer for debugging.""")
                    st.stop()
            if DEPLOYMENT_TYPE == 'Image Classification':
                pred_classname = inference_output['pred_classname']
                probability = inference_output['probability']
                caption = (f"{filename}; "
                           f"Predicted: {pred_classname}; "
                           f"Score: {probability * 100:.1f}")
                output_img = img
                channels = "BGR"
                st.image(output_img, channels=channels,
                         width=display_width, caption=caption)
            elif DEPLOYMENT_TYPE == 'Semantic Segmentation with Polygons':
                output_img = inference_output['img']
                # convert to RGB for visualizing with Matplotlib
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                channels = "RGB"

                fig = plt.figure()
                plt.subplot(121)
                plt.title("Original Image")
                plt.imshow(rgb_img)
                plt.axis('off')

                plt.subplot(122)
                plt.title("Predicted")
                plt.imshow(output_img)
                plt.axis('off')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                output_img = inference_output['img']
                channels = "RGB"
                st.image(output_img, width=display_width,
                         caption=f'Detection result for: {filename}')
            st.markdown("___")

        # Publish inference results through MQTT
        results = get_result_fn(**inference_output)
        payload = json.dumps(results)
        client.publish(topics.publish_results, payload, qos=mqtt_conf.qos)
        logger.info(f"Inference results published for {filename}")

        # publish output image through MQTT
        frame_bytes = image_to_bytes(output_img, channels)
        client.publish(topics.publish_frame[0], frame_bytes, qos=mqtt_conf.qos)
        logger.info(f"Output image published for {filename}")

        # the continuous loop should be at the end to avoid blocking others
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

    elif conf.input_type == 'Video':
        reset_camera_place.button(
            "Reset all cameras", on_click=reset_video_deployment,
            help="""Click this button if you experience any issues  \nwith the cameras
            and want to reset them.""")

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
            idx = options.index(conf.video_type)
            st.sidebar.radio(
                "Select type of video input",
                options, index=idx,
                key='video_type', help="MQTT should publish frames in bytes to the topic  \n"
                f"*{mqtt_conf.topics.recv_frame}* (topic can be changed below)",
                on_change=update_conf_and_reset_video_deploy, args=('video_type',))
        else:
            st.sidebar.markdown(
                f"**Video input type**: {conf.video_type}")

        st.sidebar.slider(
            'Width of video (for display only)', 320, 1920,
            conf.video_width, 10,
            key='video_width',
            help="This is the width of video for visualization purpose.",
            on_change=update_deploy_conf, args=('video_width',)
        )

        if conf.video_type == 'Video Camera':
            with st.sidebar.container():
                if has_access:
                    def update_and_reset_multi_cam_conf(conf_attr: str):
                        update_conf_and_reset_video_deploy(conf_attr)

                        if not conf.use_multi_cam:
                            reset_single_camera_conf()
                            return

                        logger.debug("Resetting multi-camera configuration")
                        reset_multi_camera_config()

                    if st.checkbox(
                        "Use multiple cameras", conf.use_multi_cam,
                        key='use_multi_cam', on_change=update_and_reset_multi_cam_conf,
                            args=('use_multi_cam',)):
                        conf.num_cameras = (2 if conf.num_cameras == 1
                                            else conf.num_cameras)
                        with st.form("form_num_cameras", clear_on_submit=True):
                            st.number_input(
                                "Number of cameras", 2, 10, conf.num_cameras, 1,
                                key='num_cameras')
                            st.form_submit_button(
                                "Update number of cameras",
                                on_click=update_and_reset_multi_cam_conf,
                                args=('num_cameras',))

                    # reset camera sources & titles if not same length
                    if len(conf.camera_sources) != conf.num_cameras:
                        logger.debug("Resetting multi-camera configuration")
                        reset_multi_camera_config()

                    with st.expander("Select Camera Types", expanded=True):
                        with st.form("form_camera_types", clear_on_submit=True):
                            def update_camera_types():
                                logger.info(
                                    "Reset video deployment and update camera types")
                                reset_video_deployment()
                                # reset camera_titles & camera_sources too
                                reset_multi_camera_config()

                                conf.camera_types = []
                                for i in range(conf.num_cameras):
                                    new = session_state[f'input_camera_type_{i}']
                                    conf.camera_types.append(new)

                                update_camera_sources_from_types()

                            for i in range(conf.num_cameras):
                                options = ('USB Camera', 'IP Camera')
                                idx = options.index(
                                    conf.camera_types[i])
                                st.radio(
                                    f"Type for Camera {i}", options, index=idx,
                                    key=f'input_camera_type_{i}')

                            st.form_submit_button(
                                "Update camera types", on_click=update_camera_types)

                    def reset_video_deployment_and_ports():
                        reset_video_deployment()
                        if 'working_ports' in session_state:
                            session_state.working_ports = []
                        reset_multi_camera_config()

                    num_check_ports = st.number_input(
                        "Number of ports to check for", 5, 99, 5,
                        key='input_num_check_ports',
                        help="""Number of ports to check whether they are working or not. If you are using
                        Docker on Linux, you need to specify a number at least as high as the highest number
                        of the camera *dev path*. For example, **9** if the largest *dev path* is:
                        **/dev/video9**.""")
                    st.button("Refresh camera ports", key='btn_refresh',
                              on_click=reset_video_deployment_and_ports)

                    if not session_state.working_ports:
                        with st.spinner("Checking available camera ports ..."):
                            _, working_ports = list_available_cameras(
                                num_check_ports)
                            session_state.working_ports = working_ports.copy()
                        if not working_ports:
                            st.info(
                                "No working USB camera port found.")
                            logger.info(
                                "No working USB camera port found")

                    input_num_usb_cameras = 0
                    for cam_type in conf.camera_types:
                        if cam_type == 'USB Camera':
                            input_num_usb_cameras += 1
                    num_ports = len(session_state.working_ports)
                    if num_ports < input_num_usb_cameras:
                        error_msg_place.error(
                            f"""The number of USB cameras available 
                            (**{num_ports}**) is less than the number of
                            USB cameras specified: **{input_num_usb_cameras}**."""
                        )
                        st.stop()

                    with st.expander("Notes about IP Camera Address"):
                        st.markdown(
                            """IP camera address could start with *http* or *rtsp*.
                            Most of the IP cameras have a username and password to access
                            the video. In such case, the credentials have to be provided
                            in the streaming URL as follow: 
                            **rtsp://username:password@192.168.1.64/1**""")
                else:
                    st.markdown(
                        f"Deploying with **{conf.num_cameras} camera(s)**")

                if has_access:
                    def update_camera_config():
                        logger.info(
                            "Reset video deployment and update video config")
                        reset_video_deployment()

                        if not conf.use_multi_cam:
                            # take the first input
                            conf.camera_sources = [
                                session_state['input_cam_source_0']]
                            # no need camera title/view
                            conf.camera_titles = ['']
                            return

                        new_camera_sources = []
                        new_camera_titles = []
                        for i in range(conf.num_cameras):
                            new_cam_source = session_state[f'input_cam_source_{i}']
                            new_cam_title = session_state[f'input_cam_title_{i}']
                            if new_cam_source in new_camera_sources:
                                error_msg_place.error(
                                    """Duplicated camera sources found. Each IP camera
                                    address and USB camera address must be unique!""")
                                logger.error("Duplicated camera sources found")
                                st.stop()
                            new_camera_sources.append(new_cam_source)
                            new_camera_titles.append(new_cam_title)
                        conf.camera_sources = new_camera_sources
                        conf.camera_titles = new_camera_titles

                    with st.expander("Enter Video Sources & Views", expanded=True):
                        form_col = st.container()

                    with form_col.form("form_camera_sources", clear_on_submit=True):
                        for i in range(conf.num_cameras):
                            if conf.camera_types[i] == 'USB Camera':
                                idx = session_state.working_ports.index(
                                    conf.camera_sources[i])
                                st.radio(
                                    f"Port for Camera {i}",
                                    options=session_state.working_ports,
                                    index=idx,
                                    key=f'input_cam_source_{i}')
                            else:
                                st.text_input(
                                    f"IP Camera Address {i}",
                                    value=conf.camera_sources[i],
                                    key=f'input_cam_source_{i}',
                                    placeholder='rtsp://username:password@192.168.1.64/1')

                            # Only multi-camera needs camera title/view for label checking
                            # at different views
                            if conf.use_multi_cam:
                                st.text_input(
                                    f"Title/view for camera {i}",
                                    value=conf.camera_titles[i],
                                    key=f'input_cam_title_{i}',
                                    placeholder='e.g. top (or top/left)',
                                    help="""This is required for label checking (optional) to
                                    work. If using **multiple views** for the same camera, you
                                    can separate each view with a frontslash '/' character. For
                                    example for "top" and "left" for one camera: *top/left*""")

                        submit = st.form_submit_button(
                            "Update camera config")
                    if submit:
                        # put here instead of using on_click callback to stop script here
                        # in case any error occurs
                        update_camera_config()
                        st.experimental_rerun()
                else:
                    for src, cam_type, cam_title in zip(conf.camera_sources,
                                                        conf.camera_types,
                                                        conf.camera_titles):
                        if cam_type == 'USB Camera':
                            st.markdown(f"**Port for USB Camera {i}**: {src}")
                        else:
                            st.markdown(f"**IP Camera Address {i}:** {src}")
                        if cam_title:
                            st.markdown(
                                f"**Title/view for camera {i}**: {cam_title}")

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
                conf.retention_period = retention_period

            # show CSV directory
            csv_path = deployment.get_csv_path(datetime.now())
            csv_dir = csv_path.parents[1]
            with st.sidebar.container():
                st.markdown("___")
                st.subheader("Info about saving results")
                st.markdown("**Data retention period**")
                warning_place = st.empty()
                if has_access:
                    with st.form('retention_period_form', clear_on_submit=True):
                        st.number_input("Day", 0, 1000, conf.retention_period, 1,
                                        key='day_input')
                        st.number_input("Week", 0, 10, 0, 1, key='week_input',
                                        help='7 days per week')
                        st.number_input("Month", 0, 12, 0, 1, key='month_input',
                                        help='30 days per month')
                        st.form_submit_button(
                            'Change retention period', on_click=update_retention_period)
                retention_period = conf.retention_period

                st.markdown(f"**Retention period** = {retention_period} days")
                with st.expander("CSV save file info"):
                    st.markdown(
                        f"**Inference results will be saved continuously in**: *{csv_dir}*  \n"
                        f"A new file will be created daily and files older than the "
                        f"retention period (`{retention_period} days`) will be deleted. "
                        "Be sure to click the `Pause deployment` button "
                        "to ensure the latest CSV file is saved properly if you have any "
                        "problem with opening the file.")
        elif conf.video_type == 'Uploaded Video':
            reset_single_camera_conf()

            video_file = st.sidebar.file_uploader(
                "Upload a video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
                key='video_file_uploader')
            input_video_col = st.sidebar.container()
        else:
            reset_single_camera_conf()
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
            conf.publishing = True
            session_state.refresh = True

        def stop_publish_cb(client, userdata, msg):
            logger.info("Stopping publishing ...")
            # session_state.client_connected = False
            session_state.publishing = False
            conf.publishing = False
            conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True
            # st.success("Stopped publishing")
            # cannot use this within mqtt callback
            # st.experimental_rerun()

        def start_publish_frame_cb(client, userdata, msg):
            logger.info("Start publishing frames")
            conf.publish_frame = True
            session_state.refresh = True

        def stop_publish_frame_cb(client, userdata, msg):
            logger.info("Stopping publishing frames...")
            conf.publish_frame = False
            logger.info("Stopped")
            session_state.refresh = True

        def save_frame_cb(client, userdata, msg):
            logger.debug(f'Payload received for topic "{msg.topic}"')
            # need this to access to the frame from within mqtt callback
            nonlocal output_img, channels
            save_image(output_img, saved_frame_dir,
                       channels, conf.timezone)

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
        st.sidebar.info(
            f"**MQTT broker**: localhost  \n**Port**: {mqtt_conf.port}")

        if has_access:
            # reset the list of publish_frame topics
            if len(topics.publish_frame) != conf.num_cameras:
                topics.publish_frame = [
                    f'{ORI_PUBLISH_FRAME_TOPIC}_{i}'
                    for i in range(conf.num_cameras)]

            st.sidebar.radio(
                'MQTT QoS', (0, 1, 2), mqtt_conf.qos, key='mqtt_qos',
                on_change=update_mqtt_qos)

            st.sidebar.markdown("**MQTT Topics**")
            with st.sidebar.expander("Topic Configuration", expanded=True):
                st.markdown(
                    "If you change any MQTT topic name(s), please click the **Update Config** "
                    "button to allow the changes to be made.")
                # must clear on submit to show the correct values on form
                # NOTE: the key name must be the same as the topic attribute name
                with st.form('form_mqtt_topics', clear_on_submit=True):
                    for i in range(conf.num_cameras):
                        st.text_input(
                            f'Publishing frames for camera {i} to', topics.publish_frame[i],
                            key=f'publish_frame_{i}', help="This is used to publish output "
                            "frames. Our MQTT client is not subscribed to this topic.")
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
            st.sidebar.info(f"**MQTT QoS**: {mqtt_conf.qos}")
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
        with st.sidebar.expander("Notes about deployment"):
            st.markdown(
                f"Make sure to connect to **'localhost'** broker (or IP Address of this PC) "
                f"with the correct port **{mqtt_conf.port}**. "
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
        if conf.input_type == 'Video':
            if conf.video_type == 'Video Camera':
                # only show these if a camera is not selected and not deployed yet
                if not session_state.deployed:
                    if not deploy_btn_place.button(
                        "🛠️ Deploy Model", key='btn_deploy_cam',
                            help='Deploy your model with the selected camera source'):
                        st.stop()

                    load_multi_webcams()

            elif conf.video_type == 'Uploaded Video':
                if video_file is None:
                    logger.info("Resetting video deployment")
                    # reset deployment and remove the TEMP_DIR if there's no uploaded file
                    # to ensure the app only reads the new uploaded file
                    reset_video_deployment()
                    if TEMP_DIR.exists():
                        logger.info("Removing and recreating temporary directory "
                                    "for uploaded video")
                        shutil.rmtree(TEMP_DIR)
                    os.makedirs(TEMP_DIR)
                    st.stop()

                # using str to ensure readable by st.video and cv2
                video_path = str(TEMP_DIR / video_file.name)

                if not os.path.exists(video_path):
                    # only creates it if not exists, to speed up the process when
                    # there is any widget changes besides the st.file_uploader
                    logger.info("Creating temporary video file for uploaded video at "
                                f"{video_path = }")
                    with st.spinner("Copying video to a temporary directory ..."):
                        with open(video_path, 'wb') as f:
                            f.write(video_file.getvalue())

                with input_video_col:
                    st.subheader('Input Video')
                    logger.info("Showing the uploaded video")
                    try:
                        with st.spinner("Showing the uploaded video ..."):
                            st.video(video_path)
                    except Exception as e:
                        error_msg_place.error(
                            f"Unable to read from the video file: '{video_file.name}'. "
                            "Please make sure to provide a valid video file.")
                        logger.error(f"Unable to read from the video file: "
                                     f"'{video_file.name}' with error: {e}")
                        st.stop()

                if not session_state.deployed:
                    if not deploy_btn_place.button(
                        "🛠️ Deploy Model", key='btn_deploy_uploaded_vid',
                            help='Deploy your model with the uploaded video'):
                        st.stop()

                    logger.info("Loading the uploaded video for deployment")
                    with st.spinner("Loading the video for deployment ..."):
                        try:
                            session_state.camera = cv2.VideoCapture(video_path)
                            assert session_state.camera.isOpened(), "Video is unreadable"
                        except Exception as e:
                            error_msg_place.error(f"Unable to read from the video file: "
                                                  f"'{video_file.name}'")
                            logger.error(f"Unable to read from the video file: "
                                         f"'{video_file.name}' with error: {e}")
                            reset_camera()
                            st.stop()

                        session_state.deployed = True
                        sleep(2)  # give the camera some time to sink in
                    # st.experimental_rerun()
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

            if conf.video_type == 'From MQTT':
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
                widths, heights = [width], [height]
        else:
            # stop if the "Deploy Model" button is not clicked yet
            logger.info("Model is not deployed yet")
            st.stop()

        # *********************** DOBOT arm demo ***********************
        DOBOT_TASK = dobot_demo.DobotTask.Box  # for box shapes
        # DOBOT_TASK = dobot_demo.DobotTask.P2_143  # for machine part P2/143
        # DOBOT_TASK = dobot_demo.DobotTask.P2_140  # for machine part P2/140
        # DOBOT_TASK = dobot_demo.DobotTask.DEBUG  # for debugging publishing MQTT

        # if DOBOT_TASK == dobot_demo.DobotTask.Box:
        #     VIEW_LABELS = dobot_demo.BOX_VIEW_LABELS
        # elif DOBOT_TASK == dobot_demo.DobotTask.P2_143:
        #     VIEW_LABELS = dobot_demo.P2_143_VIEW_LABELS
        # elif DOBOT_TASK == dobot_demo.DobotTask.P2_140:
        #     VIEW_LABELS = dobot_demo.P2_140_VIEW_LABELS
        # elif DOBOT_TASK == dobot_demo.DobotTask.DEBUG:
        #     VIEW_LABELS = dobot_demo.DEBUG_VIEW_LABELS

        # initialize it just in case we are not using this button
        dobot_process = None
        if st.button(
            "Move DOBOT and detect", key='btn_move_dobot',
                help='This takes awhile to start up, please be patient'):
            with st.spinner("Preparing to run dobot ..."):
                logger.info("Preparing to run dobot")
                # this will take some time to connect to the dobot api
                # and then start moving the dobot in another Process
                dobot_connnect_success, dobot_process = dobot_demo.run(
                    mqtt_conf, DOBOT_TASK)
                if not dobot_connnect_success:
                    error_msg_place.error(
                        "Failed to connect to DOBOT for demo")
                    logger.error("Failed to connect to DOBOT for demo")
                    st.stop()

        # *********************** Inspection using labels from CSV file ***********************
        csv_process = None
        if st.button(
                "Start Inspection", key='start_inspection', help='This takes awhile to start up, please be patient'):
            with st.spinner("Starting Inspection"):
                logger.info("Starting Inspection")

                # runs the function to get the labels from the csv file and use them for inspection
                csv_process_connnect_success, csv_process = csv_labels.run((
                    deployment.get_frame_save_dir('csv-labels'),))

                # Checks if the function has run
                if not csv_process_connnect_success:
                    error_msg_place.error(
                        "Failed to read from csv file")
                    logger.error("Failed to read from csv file")
                    st.stop()

        # *********************** Inspection using labels from CSV file ***********************
        node_red_process = None
        if st.button(
                "Start Node-Red Inspection", key='start_ndoe_red_inspection', help='This takes awhile to start up, please be patient'):
            with st.spinner("Starting Inspection"):
                logger.info("Starting Inspection")

                # runs the function to get the labels from the non dobot version of the node-red flow and use them for inspection
                node_red_process_connnect_success, node_red_process = Label_View.run()

                # # runs the function to get the labels from the dobot version of the node-red flow and use them for inspection
                # node_red_process_connnect_success, node_red_process = Dobot_Version.run()

                # Checks if the function has run
                if not node_red_process_connnect_success:
                    error_msg_place.error(
                        "Failed to read from Node-red file")
                    logger.error("Failed to read from Node-red file")
                    st.stop()

        # *********************** Deployment video loop ***********************
        def create_video_writer_if_not_exists(video_idx: int):
            vid_writer_key = f'vid_writer_{video_idx}'
            if session_state.get(vid_writer_key) is None:
                logger.info("Creating video file to record to")
                # NOTE: THIS VIDEO SAVE FORMAT IS VERY PLATFORM DEPENDENT
                # TODO: THE VIDEO FILE MIGHT NOT SAVE PROPERLY
                # usually either MJPG + .avi, or XVID + .mp4 or mp4v + .mp4
                FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
                now = get_now_string(timezone=conf.timezone)
                filename = f"video_{video_idx}_{now}.mp4"
                video_save_path = str(recording_dir / filename)
                # st.info(f"Video is being saved to **{video_save_path}**")
                logger.info(f"Video is being saved to '{video_save_path}'")
                # this FPS value is the FPS of the output video file,
                # note that if this value is much higher than the fps
                # during the inference time, the output video will look
                # like it's moving very very fast
                FPS = 24
                session_state[vid_writer_key] = cv2.VideoWriter(
                    video_save_path, FOURCC, FPS,
                    (widths[video_idx], heights[video_idx]), True)

        publish_place = st.sidebar.empty()
        option_col1, option_col2, option_col3, option_col4 = st.columns(4)
        video_options_col = st.container()
        video_col_1, video_col_2 = st.columns(2)
        video_title_place = {}
        msg_place = {}
        output_video_place = {}
        fps_place = {}
        result_place = {}
        widths = []
        heights = []

        for i, col in zip(range(conf.num_cameras), cycle((video_col_1, video_col_2))):
            with col:
                video_title_place[i] = st.empty()
                msg_place[i] = st.empty()
                output_video_place[i] = st.empty()
                # fps_col, width_col, height_col = st.columns(3)
                fps_col = st.container()
                result_place[i] = st.empty()
                st.markdown("___")

            if conf.video_type == 'Video Camera':
                cam_title = conf.camera_titles[i]
                video_title_place[i].subheader(
                    f"Video Camera {i}: {cam_title}")
                cam_key = conf.camera_keys[i]
                stream = session_state[cam_key].stream
                deploy_status_place.info(
                    "**Status**: Deployed for camera inputs")
            else:
                video_title_place[i].subheader("Output Video")
                # uploaded video just has one camera
                stream = session_state.camera
                deploy_status_place.info(
                    "**Status**: Deployed for uploaded video")

            width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(stream.get(cv2.CAP_PROP_FPS))
            logger.info(
                f"Video properties for camera {i}: "
                f"{width = }, {height = }, {fps_input = }")
            # store them to use for VideoWriter later
            widths.append(width)
            heights.append(height)

            with fps_col:
                st.markdown(f"**Frame Width**: {width}")
                st.markdown(f"**Frame Height**: {height}")
                # st.markdown("**Frame Rate**")
                fps_place[i] = st.markdown("0")
            # with width_col:
            #     st.markdown("**Frame Width**")
            #     st.markdown(kpi_format(width), unsafe_allow_html=True)
            # with height_col:
            #     st.markdown("**Frame Height**")
            #     st.markdown(kpi_format(height), unsafe_allow_html=True)

        show_video = option_col1.checkbox(
            'Show video', value=True, key='show_video')
        if DEPLOYMENT_TYPE == 'Object Detection with Bounding Boxes':
            get_bbox_coords = option_col4.checkbox("Obtain bounding box coordinates",
                                                   key='get_bbox_coords')
        draw_result = option_col2.checkbox(
            "Draw labels", value=True, key='draw_result')
        show_labels = option_col3.checkbox(
            "Show the detected labels", value=True)

        with video_options_col:
            if show_labels:
                for i in range(conf.num_cameras):
                    with result_place[i].container():
                        st.markdown("**Detected Results**")
                        st.markdown("Coming up")

            def update_record(is_recording: bool):
                session_state.record = is_recording

            if not session_state.record:
                st.button('Start recording', key='btn_start_record',
                          help=f"Videos are saved in *{recording_dir}*",
                          on_click=update_record, args=(True,))
            else:
                st.button("Stop recording and save the video",
                          key='btn_stop_and_save_vid',
                          on_click=update_record, args=(False,))
            st.markdown("___")

            msg_place['top'] = st.empty()  # for general messages

        if has_access:
            def update_publishing_conf(is_publishing: bool):
                conf.publishing = is_publishing
                session_state.publishing = is_publishing
                # also change whether publishing frame or not
                conf.publish_frame = is_publishing

            def update_publish_frame_conf(publish_frame: bool):
                conf.publish_frame = publish_frame

            if session_state.publishing:
                # using buttons to allow the widget to change after rerun
                # whereas checkbox does not change after rerun
                publish_place.button("Stop publishing results", key='btn_stop_pub',
                                     on_click=update_publishing_conf, args=(False,))
            else:
                publish_place.button("Start publishing results", key='btn_start_pub',
                                     on_click=update_publishing_conf, args=(True,))

            if conf.publish_frame:
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
            session_state.publishing = conf.publishing
            if session_state.publishing:
                st.sidebar.markdown("Currently is publishing results to the topic: "
                                    f"*{topics.publish_results}*")
            else:
                st.sidebar.markdown(
                    "Currently is not publishing any results through MQTT.")

            if conf.publish_frame:
                st.sidebar.markdown(
                    "Currently is publishing output frames for each camera to the topics:")
                for i in range(conf.num_cameras):
                    st.sidebar.markdown(f"{i}. *{topics.publish_frame}*")
            else:
                st.sidebar.markdown(
                    "Currently is not publishing any output frames through MQTT.")

        # prepare variables for the video deployment loop
        timezone = conf.timezone
        inference_pipeline = deployment.get_inference_pipeline(
            draw_result=draw_result, **pipeline_kwargs)

        get_result_fn = get_result_postprocessor()

        publish_func = partial(client.publish,
                               topics.publish_results, qos=mqtt_conf.qos)
        publish_frame_funcs = []
        for i in range(conf.num_cameras):
            publish_frame_funcs.append(
                partial(client.publish,
                        topics.publish_frame[i], qos=mqtt_conf.qos))

        starting_time = datetime.now()
        logger.info(f"Starting RAM usage: {get_ram_usage()} %")
        # use this to check when to clear memory
        memory_clear_start = perf_counter()
        # clear memory every 15 minutes (900 seconds)
        MEMORY_CLEAR_INTERVAL = 900
        csv_path = deployment.get_csv_path(starting_time)
        csv_dir = csv_path.parent
        if not csv_dir.exists():
            os.makedirs(csv_dir)
        logger.info(f'Operation begins at: {starting_time.isoformat()}')
        logger.info(f'Inference results will be saved in {csv_dir}')
        if conf.video_type == 'Video Camera':
            video_type = 0
        elif conf.video_type == 'From MQTT':
            video_type = 1
        else:
            # Uploaded Video
            video_type = 2
        use_multi_cam = conf.use_multi_cam
        camera_keys = conf.camera_keys
        if use_multi_cam:
            cam_titles = conf.camera_titles
            camera_view2idx = deployment.get_camera_views_from_titles(
                cam_titles)
        else:
            cam_title = ''
        display_width = conf.video_width
        publish_frame = conf.publish_frame
        first_csv_save = True

        # start the video deployment loop
        for i in cycle(range(conf.num_cameras)):
            start_time = perf_counter()

            # clear memory at an interval
            if (start_time - memory_clear_start) > MEMORY_CLEAR_INTERVAL:
                # restart clear interval
                memory_clear_start = perf_counter()
                logger.info(
                    f"Clearing unused memory [Current RAM usage: {get_ram_usage()}%]")
                with msg_place['top'].container():
                    with st.spinner("Clearing memory ..."):
                        gc.collect()
                # seems like must rerun only can clear properly
                st.experimental_rerun()

            if session_state.refresh:
                # refresh page once to refresh the widgets
                logger.debug("REFRESHING PAGEE")
                session_state.refresh = False
                st.experimental_rerun()

            if video_type == 0:
                # USB Camera or IP Camera
                frame = session_state[camera_keys[i]].read()
            elif video_type == 1:
                # from MQTT
                frame = image_from_buffer(session_state.mqtt_recv_frame)
            else:
                # Uploaded Video, read with cv2.VideoCapture instead of WebcamVideoStream
                ret, frame = session_state.camera.read()
                if not ret:
                    break

            if frame is None:
                msg_place[i].error("""Unable to read camera frame from camera,
                    restarting the cameras ...""")
                logger.error(f"Error reading frame from Camera {i}, "
                             "trying to restart the cameras now ...")
                reset_camera()
                with msg_place['top'].container():
                    while True:
                        try:
                            load_multi_webcams(stop_on_error=False)
                        except CameraFailError as e:
                            msg_place[i].error(
                                f"Reopen failed for camera {i}. Retrying in 3 seconds ...")
                            logger.error(
                                f"Reopen failed: {e}. Retrying in 3 seconds ...")

                            reset_camera()
                            sleep(3)
                            continue
                        else:
                            break
                st.experimental_rerun()

            if use_multi_cam:
                cam_title = cam_titles[i]

            # frame.flags.writeable = True  # might need this?
            # run inference on the frame
            inference_output = inference_pipeline(frame)
            output_img = inference_output['img']
            channels = inference_output['channels']
            results = get_result_fn(**inference_output, camera_title=cam_title)

            if show_video:
                output_video_place[i].image(output_img, channels=channels,
                                            width=display_width)

            vid_writer_key = f'vid_writer_{i}'
            if session_state.record:
                # need to be within the video loop to ensure we also get the latest
                #  session_state updates from MQTT callback
                msg_place[i].info("Recording ...")
                create_video_writer_if_not_exists(i)
                if channels == 'RGB':
                    # cv2.VideoWriter needs BGR format
                    out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                else:
                    out = output_img
                session_state[vid_writer_key].write(out)
            else:
                if session_state.get(vid_writer_key) is not None:
                    msg_place[i].empty()
                    logger.info("Saving recorded file")
                    # must release to close the video file
                    session_state[vid_writer_key].release()
                    del session_state[vid_writer_key]

            # count FPS
            fps = 1 / (perf_counter() - start_time)

            # fps_place[i].markdown(kpi_format(int(fps)),
            #                       unsafe_allow_html=True)
            fps_place[i].markdown(f"**Frame Rate**: {int(fps)}")

            if show_labels:
                result_place[i].table(results)

            if publish_frame:
                frame_bytes = image_to_bytes(output_img, channels)
                publish_frame_funcs[i](frame_bytes)

            # NOTE: this session_state is either used for DOBOT arm for
            # object detection demo to detect different labels at different views,
            # or can also use for label checking when MQTT message is received
            if session_state.check_labels:
                # session_state.check_labels should be Tuple[str, List[str]]
                # generated from validate_received_label_msg() in dobot_view_cb()
                view, required_labels = session_state.check_labels
                view: str = view.lower()

                if view == 'end':
                    # clear the messages if the robot motion has ended
                    for p in msg_place.values():
                        p.empty()
                    # and reset back to None
                    session_state.check_labels = None
                    logger.info("Label checking process has finished")
                    if csv_process is not None:
                        # gracefully kill the dobot_demo's Process
                        csv_process.kill()

                    # NOTE: infinite loop for debugging purposes
                    # dobot_connnect_success, dobot_process = dobot_demo.run(
                    #     mqtt_conf, DOBOT_TASK)
                    # if not dobot_connnect_success:
                    #     error_msg_place.error(
                    #         "Failed to connect to DOBOT for demo")
                    #     logger.error("Failed to connect to DOBOT for demo")
                    continue

                if use_multi_cam:
                    # we only let user enter the views if there is more than 1 camera
                    src_idx = camera_view2idx.get(view)
                    if src_idx is None:
                        logger.error(
                            f"Cannot find '{view}' view from the entered camera views. "
                            "Please try updating the camera views to follow MQTT message "
                            "(or vice versa).")
                        msg_place['top'].error(
                            f"Cannot find **'{view}'** view from the entered camera views. "
                            "Please try updating the camera views to follow MQTT message "
                            "(or vice versa).")
                        session_state.check_labels = None
                        continue
                    # continue until the current loop is for the correct view
                    if src_idx != i:
                        continue
                else:
                    src_idx = 0

                # required_label_cnts = VIEW_LABELS[view]
                # if dobot_demo.check_result_labels(results, required_label_cnts):
                if deployment.check_labels(results, required_labels, DEPLOYMENT_TYPE):
                    logger.info(f"All labels present at '{view}' view")
                    msg_place[src_idx].success(
                        f"### {view.capitalize()} view: OK")
                    client.publish(
                        mqtt_conf.topics.send_result_okng, payload="OK")
                    client.publish(
                        mqtt_conf.topics.current_view, payload=view)
                    client.publish(
                        mqtt_conf.topics.detected_labels, payload=results)
                else:
                    logger.warning("Required labels are not detected at "
                                   f"'{view}' view")
                    msg_place[src_idx].error(
                        f"### {view.capitalize()} view: NG")
                    client.publish(
                        mqtt_conf.topics.send_result_okng, payload="NG")
                    client.publish(
                        mqtt_conf.topics.current_view, payload=view)
                    client.publish(
                        mqtt_conf.topics.detected_labels, payload=results)
                    save_image(output_img, ng_frame_dir,
                               channels, timezone=timezone, prefix=view)
                    logger.info(f"NG image saved successfully")

                # set this to None to ONLY CHECK FOR ONCE for the same view
                session_state.check_labels = None

            if not results:
                continue

            if session_state.publishing:
                payload = json.dumps(results)
                publish_func(payload=payload)

            # save results to CSV file only if using video camera
            if video_type != 0:
                continue

            if first_csv_save:
                first_csv_save = False
                create_csv_file_and_writer(csv_path, results)
            now = datetime.now()
            today = now.date()
            if today > session_state.today:
                session_state.csv_file.close()

                deployment.delete_old_csv_files(
                    retention_period)

                csv_path = deployment.get_csv_path(now)
                create_csv_file_and_writer(csv_path, results)
                session_state.today = today
            else:
                for row in results:
                    session_state.csv_writer.writerow(row)

            # This below does not seem to work properly
            # if fps > max_allowed_fps:
            #     sleep(1 / (fps - max_allowed_fps))

        # clean up everything if it's an uploaded video
        reset_video_deployment()

        if TEMP_DIR.exists():
            logger.debug("Removing temporary directory")
            shutil.rmtree(TEMP_DIR)

        logger.info("Inference done for uploaded video")
        msg_place[i].info("Inference done for uploaded video.")


def main():
    # False for debugging
    index(RELEASE=False)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
