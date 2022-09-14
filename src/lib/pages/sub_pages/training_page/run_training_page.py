""" 
Title: Training Page
Date: 29/9/2021 
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
import gc
import os
import shutil
import sys
from pathlib import Path
from time import sleep

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

import tensorflow as tf

# >>>> **************** TEMP (for debugging) **************** >>>
# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

# # Set to wide page layout for debugging on this page
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>> **************** TEMP **************** >>>

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from machine_learning.trainer import Trainer
from machine_learning.command_utils import kill_process, kill_tensorboard, run_tensorboard
from machine_learning.visuals import pretty_format_param
from project.project_management import Project
from training.training_management import NewTrainingPagination, Training, TrainingPagination
from user.user_management import User
from deployment.deployment_management import Deployment


def index(RELEASE=True):
    logger.debug("Navigator: At run_training_page.py")

    # ****************** TEST ******************************
    if not RELEASE:

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        # for Anson: 4 for TFOD, 9 for img classif, 30 for segmentation
        # uploaded pet segmentation: 96
        # uploaded face detection: 111
        # dogs vs cats classification - small (uploaded): 98
        project_id_tmp = 97
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            # for Anson: 2 for TFOD, 17 for img classif, 18 for segmentation
            # uploaded pet segmentation: 20
            # uploaded face detection: 32
            # dogs vs cats classification - small (uploaded): 42
            training_id_tmp = 22
            session_state.new_training = Training(training_id_tmp,
                                                  session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
    # ****************** TEST END ******************************

    training: Training = session_state.new_training
    project: Project = session_state.project
    trainer: Trainer = session_state.get('trainer')
    training_paths = training.get_paths()

    if 'deployment' in session_state:
        logger.info("Resetting deployment page to avoid issues with training")
        Deployment.reset_deployment_page()

    def reset_trainer():
        if 'trainer' in session_state:
            del session_state['trainer']

    def initialize_trainer(reset: bool = False):
        if reset:
            reset_trainer()

        if 'trainer' not in session_state:
            # store the trainer to use for training, inference and deployment
            with st.spinner("Initializing trainer ..."):
                logger.info("Initializing trainer")
                session_state.trainer = Trainer(project, training)
                nonlocal trainer
                trainer = session_state.trainer

    def show_edit_training_btn():
        def back_train_info_page():
            # reset it in case the user decided to change any info
            reset_trainer()
            session_state.new_training_pagination = NewTrainingPagination.InfoDataset

        st.button('⚙️ Edit Training Info', key='btn_edit_train_info',
                  on_click=back_train_info_page)

    def show_edit_model_btn():
        def back_model_page_cb():
            reset_trainer()
            session_state.new_training_pagination = NewTrainingPagination.Model

        st.button('⚙️ Edit Model Info', key='btn_edit_model_info',
                  on_click=back_model_page_cb)

    st.markdown("### Training Info:")

    dataset_chosen = training.dataset_chosen
    if not dataset_chosen:
        st.warning("""This training session **does not have any training dataset** 
        chosen yet. Please proceed to the training info page to select it.""")
        show_edit_training_btn()
        st.stop()

    if len(dataset_chosen) == 1:
        dataset_chosen_str = dataset_chosen[0]
    else:
        dataset_chosen_str = []
        for idx, data in enumerate(dataset_chosen):
            dataset_chosen_str.append(f"**{idx+1}**. {data}")
        dataset_chosen_str = '; '.join(dataset_chosen_str)
    partition_ratio = training.partition_ratio
    st.info(f"""
    **Training Session Name**: {training.name}  \n
    **Training Description**: {training.desc}  \n
    **Dataset List**: {dataset_chosen_str}  \n
    **Partition Ratio**: training : validation : test -> 
    {partition_ratio['train']} : {partition_ratio['eval']} : {partition_ratio['test']}  \n
    **Partition Size**: training : validation : test -> 
    {training.partition_size['train']} :
    {training.partition_size['eval']} :
    {training.partition_size['test']}  \n
    **Selected Model Name**: {training.attached_model.name}  \n
    **Model Name**: {training.training_model.name}  \n
    **Model Description**: {training.training_model.desc}
    """)

    train_info_btn, model_info_btn, _ = st.columns([2, 1, 4])

    with train_info_btn:
        show_edit_training_btn()

    with model_info_btn:
        show_edit_model_btn()

    if training.is_started:
        st.warning("✏️ **NOTE**: Only edit the model selection"
                   " if you want to re-train your model! Otherwise the information"
                   " stored in database would not be correct for the current trained model.")

    if "CenterNet MobileNetV2" in training.attached_model.name:
        st.warning(
            """Please change your model from CenterNet MobileNetV2 to 
        something else as it experiences weird issues currently.""")
        st.stop()

    # ******************************** CONFIG INFO ********************************
    train_config_col, aug_config_col = st.columns([1, 1])

    with train_config_col:
        config_info = pretty_format_param(training.training_param_dict)
        st.markdown('### Training Config:')
        st.info(config_info)

        def back_config_page_cb():
            reset_trainer()
            session_state.new_training_pagination = NewTrainingPagination.TrainingConfig

        st.button('⚙️ Edit Training Config', key='btn_edit_config',
                  on_click=back_config_page_cb)

    with aug_config_col:
        # TODO: confirm that this works for image classification and segmentation
        st.markdown('### Augmentation Config:')
        augmentation_config = training.augmentation_config
        if augmentation_config.exists():
            aug_config_info = pretty_format_param(
                augmentation_config.to_dict())
            st.info(aug_config_info)
        else:
            st.info("No augmentation config selected.")

        def back_aug_config_page_cb():
            reset_trainer()
            session_state.new_training_pagination = NewTrainingPagination.AugmentationConfig

        st.button('⚙️ Edit Augmentation Config', key='btn_edit_aug_config',
                  on_click=back_aug_config_page_cb)

    if training.is_started:
        st.warning(
            "✏️ **NOTE**: Only edit training/augmentation config if you want to "
            "re-train or continue training!")

    st.markdown("___")

    if tf.config.list_physical_devices('GPU'):
        using_gpu = True
    else:
        using_gpu = False
        st.warning("""**WARNING**: You don't have access to GPU. Training will
        take much longer time to complete without GPU. Inference time will
        also be slower.""")

    # ******************************* START TRAINING *******************************
    train_btn_place = st.empty()
    retrain_place = st.empty()
    message_place = st.empty()  # for warning messages
    result_place = st.empty()

    def start_training_callback(is_resume=False, train_one_batch=False):
        if not is_resume:
            root = training_paths['ROOT']
            if root.exists():
                logger.info(f"Removing existing training directory {root}")
                shutil.rmtree(root)

        logger.info("Creating training directories if not exists ...")
        training.initialise_training_folder()

        # moved stop button into callback function to only show when training is started
        btn_stop_col, _ = st.columns([1, 1])

        def stop_run_training():
            # this process_id is from `run_command()` or `run_command_update_metrics()`
            if session_state.get('process_id'):
                with btn_stop_col:
                    with st.spinner("Stopping the training process ..."):
                        logger.info("Stopping the training process")
                        kill_process(session_state.process_id)
                    del session_state.process_id
                    st.warning("Training stopped!")
            sleep(2)

        with btn_stop_col:
            st.button("⛔ Stop Training", key='btn_stop_training',
                      on_click=stop_run_training,
                      help="This takes a short while to stop the training")
            st.warning('✏️ **NOTE**: If you click this button, '
                       'the latest progress might not be saved.')

        logdir = training_paths['tensorboard_logdir']

        with st.spinner("Loading TensorBoard ..."):
            st.markdown("Refresh the Tensorboard by clicking the refresh "
                        "icon during training to see the progress:")
            # need to sleep for 3 seconds, otherwise the TensorBoard might accidentally
            #  load the previous checkpoints
            sleep(3)
            run_tensorboard(logdir)

        with st.spinner("Exporting tasks for training ..."):
            logger.info("Exporting tasks for training ...")
            project.export_tasks(for_training_id=training.id)

        # reset to make sure everything is new
        initialize_trainer(reset=True)
        # start training, set `stdout_output` to True to print the logging outputs generated
        #  from the TFOD scripts; set to False to avoid clutterring the console outputs
        trainer.train(
            is_resume, stdout_output=True, train_one_batch=train_one_batch)

        if 'start_idx' in session_state:
            # this is to avoid problems in the evaluation functions
            del session_state['start_idx']

        def refresh_page():
            sleep(0.5)

        st.button("Refresh Training Page", key='btn_refresh_page',
                  help="Refresh this page to show all the results",
                  on_click=refresh_page)

    if not training.is_started:
        with train_btn_place.container():
            if project.deployment_type != 'Object Detection with Bounding Boxes':
                train_one_batch = st.button(
                    "⚡ Test training one batch", key='btn_train_one_batch',
                    help="""This is just a test run to see whether the model can 
                    perform well on only one batch of data (more specifically,
                    whether it's capable enough to overfit one batch of data).
                    This is also a good way to check whether there is any problem
                    with the dataset, the model, or the training pipeline.""")
                if train_one_batch:
                    start_training_callback(train_one_batch=True)

            start_train = st.button(
                "⚡ Start training", key='btn_start_training',
                help="If you experience any error related to 'Memory' or 'Resources',  \n"
                "please try to reduce the `Batch Size` before training again.")
            if start_train:
                # must do it this way instead of using callback on button
                #  to properly show the training progress below the rendered widgets
                start_training_callback()
    else:
        with train_btn_place.container():
            st.header("Training results:")
            show_tb = st.button("📈 Show TensorBoard",
                                key='btn_show_tensorboard')
            if show_tb:
                with st.spinner("Loading Tensorboard ..."):
                    logdir = training_paths['tensorboard_logdir']
                    run_tensorboard(logdir)

            clone_col, download_col = st.columns([1, 1])

            def clone_train_session():
                training.clone_training_session()

                # get the id before resetting
                cloned_train_id = training.id
                # resetting the training page stuff to avoid memory issues
                training.reset_training_page()
                tf.keras.backend.clear_session()
                gc.collect()

                # create the session_state again after reset
                session_state.new_training = Training(
                    cloned_train_id, session_state.project)

                # set all the submissions as True to allow proper updates instead of inserting info into DB
                for page in session_state.new_training.has_submitted:
                    session_state.new_training.has_submitted[page] = True

                session_state.training_pagination = TrainingPagination.Existing

                # must go back to the InfoDataset page to allow user to
                # update the temporarily created names if necessary
                session_state.new_training_pagination = NewTrainingPagination.InfoDataset

            with clone_col:
                st.button("📋 Clone the Training Session",
                          key='btn_clone_session', on_click=clone_train_session)
                st.warning('✏️ **NOTE**: This will create a new training session, '
                           'while retaining all the current configuration, '
                           'to allow you to quickly train another model for benchmarking.')

            with download_col:
                initialize_trainer()
                exist_dict = trainer.check_model_exists()

                if exist_dict['model_tarfile']:
                    def show_download_msg():
                        place = st.empty()
                        place.warning("Downloading Model ...")
                        sleep(2)
                        place.empty()

                        # remove exported directory after downloaded
                        # export_dir = training_paths['export']
                        # if export_dir.exists():
                        #     shutil.rmtree(export_dir)

                        # remove the tarfile after downloaded
                        if model_tarfile_path.exists():
                            os.remove(model_tarfile_path)

                    model_tarfile_path = training_paths['model_tarfile']
                    with st.spinner("Preparing model to be downloaded ..."):
                        with model_tarfile_path.open(mode="rb") as fp:
                            st.download_button(
                                label="📁 Download Trained Model",
                                data=fp,
                                file_name=model_tarfile_path.name,
                                mime="application/octet-stream",
                                key="btn_download_model",
                                on_click=show_download_msg,
                            )
                        st.success('✏️ Model is successfully archived! This may take awhile'
                                   ' to download, depending on the file size of the trained '
                                   'model.')
                        # if training.deployment_type == 'Object Detection with Bounding Boxes':
                        #     st.warning("""The exported object detection model will also be
                        #     removed after you have downloaded it. You can try checking the
                        #     evaluation result below after exporting, because it will load
                        #     from the exported SavedModel format instead of checkpoint, the
                        #     results could be different.""")
                # only show Export button if model checkpoint/weights file is found
                elif exist_dict['ckpt']:
                    def export_callback():
                        try:
                            trainer.export_model(re_export=False)
                        except Exception as e:
                            st.error("""Some error has occurred when exporting,
                                please try re-training again""")
                            logger.error(f"Error exporting model: {e}")
                            if os.getenv('DEBUG', '1') == '1':
                                st.exception(e)
                            st.stop()
                    if training.deployment_type == 'Object Detection with Bounding Boxes':
                        label = "📁 Export TensorFlow SavedModel"
                    else:
                        label = "📁 Export TensorFlow Keras Model"
                    export = st.button(label, key='btn_export_model')
                    st.warning('✏️ **NOTE**: This may take awhile to export, '
                               'depending on the size of the trained model.')
                    if export:
                        export_callback()
                        st.experimental_rerun()

        with retrain_place.container():
            retrain_col_1, resume_train_col = st.columns([1, 1])
            with retrain_col_1:
                retrain = st.button("⚡ Re-train", key='btn_retrain')
                st.warning('✏️ **NOTE**: If you re-train your model, '
                           'all the existing model data will be overwritten.')

            with resume_train_col:
                resume_train = st.button(
                    "⚡ Continue training", key='btn_resume_train')
                if training.deployment_type == 'Object Detection with Bounding Boxes':
                    num_train_steps = training.training_param_dict['num_train_steps']
                    current_step = training.progress['Step']
                    total_steps = current_step + num_train_steps
                    value_msg = f"""**{num_train_steps}** steps. Current step is 
                    **{current_step}**, i.e. after training, total of **{total_steps}**
                    steps"""
                else:
                    epochs = training.training_param_dict['num_epochs']
                    current_epoch = training.progress['Epoch']
                    total_epochs = current_epoch + epochs
                    value_msg = f"""**{epochs}** epochs. Current epoch is
                    **{current_epoch}**, i.e. after training, total of
                    **{total_epochs}** epochs"""
                st.warning(
                    f"""✏️ This will continue training your model based on the training
                    config specified: {value_msg}.""")

        with result_place.container():
            metric_col_1, _ = st.columns([1, 1])
            with metric_col_1:
                metrics = pretty_format_param(
                    training.training_model.metrics)
                st.markdown("#### Final Metrics:")
                progress_text = pretty_format_param(
                    training.progress, st_newlines=False, bold_name=True)
                st.markdown(f"Latest progress at {progress_text}")
                st.info(metrics)

                if training.deployment_type == 'Object Detection with Bounding Boxes':
                    with st.expander("Notion about the metrics"):
                        st.markdown(
                            "From [TFOD docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model):  \n")
                        st.markdown("""Following what people have said online,
                        it seems that it is advisable to allow your model to reach a `Total loss`
                        of **at least 2** (ideally 1 and lower) if you want to achieve “fair” 
                        detection results. Obviously, lower `Total loss` is better, however very 
                        low `Total loss` should be avoided, as the model may end up overfitting 
                        the dataset, meaning that it will perform poorly when applied to images
                        outside the dataset it has been trained on. To monitor `Total loss`, as well
                        as a number of other metrics even while your model is training, have a look at 
                        the **TensorBoard** above.""")

            if training_paths['models'].exists():
                initialize_trainer()
                st.markdown("___")
                st.header("Evaluation results:")

                if not using_gpu:
                    st.warning("""**WARNING**: You don't have access to GPU. Inference time
                    will be slower depending on the capability of your CPU and system.""")

                try:
                    trainer.evaluate()
                except Exception as e:
                    if os.getenv('DEBUG', '1') == '1':
                        st.exception(e)
                    st.error(
                        "Some error has occurred. Please try checking the terminal "
                        "output for the error. Or just try pressing *'R'* to refresh the page. "
                        "If it still doesn't work, please try training the model again.")
                    logger.error(f"Error evaluating: {e}")
            else:
                logger.info(f"Model {training.training_model_id} "
                            "is not exported yet. Skipping evaluation")

        if retrain or resume_train:
            # clear out the unnecessary placeholders
            train_btn_place.empty()
            result_place.empty()
            retrain_place.empty()

            is_resume = False if retrain else True
            with retrain_place.container():
                start_training_callback(is_resume)

    # st.write("vars(training)")
    # st.write(vars(training))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        # This is set to False for debugging purposes
        # when running Streamlit directly from this page
        RELEASE = False

        # run this page in debugging mode
        index(RELEASE)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
