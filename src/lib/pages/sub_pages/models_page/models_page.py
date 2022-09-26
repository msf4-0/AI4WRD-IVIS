"""
Title: Models Page (Index)
Date: 5/9/2021
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
import shutil
import sys
from copy import deepcopy

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.log import logger  # logger
from path_desc import chdir_root
from data_manager.database_manager import init_connection
from data_manager.data_table_component.data_table import data_table
from data_manager.database_manager import init_connection
from pages.sub_pages.models_page.models_subpages.user_model_upload import user_model_upload_page
from project.project_management import Project
from training.model_management import Model, ModelType, ModelsPagination, NewModel, get_trained_models_df
from training.training_management import (NewTraining, NewTrainingPagination,
                                          NewTrainingSubmissionHandlers,
                                          Training)
from user.user_management import User
from training.utils import get_pretrained_model_details, get_segmentation_model_func2params
from machine_learning.visuals import pretty_format_param

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])
# <<<< Variable Declaration <<<<


def existing_models():
    DEPLOYMENT_TYPE = session_state.project.deployment_type

    if isinstance(session_state.new_training, NewTraining):
        # done creating the info in database, thus converting it
        # to a Training instance
        session_state.new_training = Training.from_new_training(
            session_state.new_training, session_state.project)
        logger.debug("Converted NewTraining to Training instance")

    training: Training = session_state.new_training

    existing_models, existing_models_column_names = deepcopy(Model.query_model_table(
        for_data_table=True,
        return_dict=True,
        deployment_type=DEPLOYMENT_TYPE))

    # st.write(vars(session_state.training))

    # ************************* SESSION STATE ************************************************
    # if "existing_models_table" not in session_state:
    #     session_state.existing_models_table = ''

    # ************************* SESSION STATE ************************************************

    # ************************ COLUMN PLACEHOLDER ********************************************
    st.write(f"**Step 1: Select a pretrained Deep Learning model "
             "from the table to be used for training, "
             "or you may also choose to upload a model**")
    to_model_upload_page_button_place = st.empty()

    # - Removed the columns arrangement
    # main_col1, main_col2 = st.columns([3, 1])
    # ************************ COLUMN PLACEHOLDER ********************************************

    def to_model_upload_page():
        session_state.models_pagination = ModelsPagination.ModelUpload

        # if "existing_models_table" in session_state:
        #     del session_state.existing_models_table

    to_model_upload_page_button_place.button(label="Upload Deep Learning Model",
                                             key="upload_new_model",
                                             on_click=to_model_upload_page)
    # **************** DATA TABLE COLUMN CONFIG *********************************************************
    existing_models_columns = [
        {
            'field': "id",
            'headerName': "ID",
            'headerAlign': "center",
            'align': "center",
            'flex': 50,
            'hideSortIcons': True,

        },
        {
            'field': "Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Framework",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Model Type",
            'headerAlign': "center",
            'align': "center",
            'flex': 120,
            'hideSortIcons': True,
        },
        {
            'field': "Training Name",
            'headerAlign': "center",
            'align': "center",
            'flex': 150,
            'hideSortIcons': False,
        },
        {
            'field': "Date/Time",
            'headerAlign': "center",
            'align': "center",
            'flex': 100,
            'hideSortIcons': True,
            'type': 'date',
        },
    ]
    # ******************************* DATA TABLE *******************************

    # CALLBACK >>>> Returns model_id -> store inside model class

    # def instantiate_model():
    #     model_df_row = Model.filtered_models_dataframe(models=existing_models,
    #                                                    # care the name of the column is capitalized
    #                                                    dataframe_col="Name",
    #                                                    #    filter_value=session_state.existing_models_table[0],
    #                                                    filter_value=session_state.existing_models_table,
    #                                                    column_names=existing_models_column_names)

    #     # Instantiate Attached Model in `training` object
    #     training.attached_model = Model(
    #         model_row=model_df_row[0])

    # with main_col1.container():
    # not using `data_table` to avoid unnecessary complexity
    # data_table(rows=existing_models,
    #            columns=existing_models_columns,
    #            checkbox=False,
    #            key='existing_models_table', on_change=instantiate_model)

    # header
    st.markdown(f"### Models Selection")

    if training.has_submitted[NewTrainingPagination.Model]:
        if training.attached_model.is_not_pretrained:
            model_type_constant = training.attached_model.model_type_constant
            if model_type_constant == ModelType.UserUpload:
                model_label = 'Selected Uploaded Model'
            else:
                model_label = 'Selected Project Model'
        else:
            model_label = 'Selected Pre-trained Model'
        selected_model_name = training.attached_model.name
        current_name = training.training_model.name
        current_desc = training.training_model.desc
        st.info(f"""
        ### Previously submitted model information:
        **{model_label}**: {selected_model_name}  \n
        **Training Model Name**: {current_name}  \n
        **Training Description**: {current_desc}  \n
        """)

    if training.attached_model is not None:
        model_type_constant = training.attached_model.model_type_constant
        if model_type_constant == ModelType.PreTrained:
            model_type_idx = 0
        elif model_type_constant == ModelType.ProjectTrained:
            model_type_idx = 1
        else:
            model_type_idx = 2
    else:
        model_type_idx = 0

    options = ('Pre-trained Model', 'Project Model', 'User-Uploaded Model')
    selected_model_type = st.radio(
        'Select the type of model', options,
        index=model_type_idx,
        key='selected_model_type',
        help=('Project Model is a model trained in our application. '
              'Recommended to start with a Pre-trained Model.'))

    if selected_model_type != 'Pre-trained Model':
        if selected_model_type == 'User-Uploaded Model':
            model_type_constant = ModelType.UserUpload
        else:
            model_type_constant = ModelType.ProjectTrained
        trained_models_df = get_trained_models_df(
            existing_models, existing_models_column_names, model_type_constant)
        if trained_models_df.empty:
            st.warning(f"There is currently no {selected_model_type} for the current "
                       "deployment type yet. Please choose an existing pre-trained models "
                       "(recommended), or upload your own model.")
            st.stop()

        trained_models_df['Metrics'] = trained_models_df['Metrics'].apply(
            pretty_format_param, st_newlines=False, bold_name=False)

        current_training_model_id = training.training_model.id
        # need to check with int because it could be a randomly initialized ID for NewModel
        if isinstance(current_training_model_id, int) and \
                model_type_constant == ModelType.ProjectTrained:
            logger.debug(f"{current_training_model_id = }")
            # remove the training model from the DataFrame if it's the current
            # training_model used for training, otherwise it makes no sense
            current_model_idx = int(
                trained_models_df.loc[
                    trained_models_df['id'] == current_training_model_id
                ].index[0]
            )
            trained_models_df = trained_models_df.drop(
                current_model_idx).reset_index(drop=True)

        st.markdown(f"**List of {selected_model_type}s:**")
        st.dataframe(trained_models_df, width=1920)

        # get the index of submitted model or just set to 0
        if training.attached_model is not None \
                and training.attached_model.model_type_constant == model_type_constant:
            trained_model_name = training.attached_model.name
            trained_model_idx = int(trained_models_df.query(
                'Name == @trained_model_name').index[0]
            )
        else:
            trained_model_idx = 0
        logger.debug(f"{trained_model_idx = }")

        # NOTE: this selectbox must not use the same key with the pretrained model selectbox
        # or weird errors will occur
        selected_model_name = st.selectbox(
            f"Please select a {selected_model_type}",
            options=trained_models_df['Name'],
            index=trained_model_idx,
            key=f"selected_{model_type_constant.name}",
        )
    else:
        logger.debug(f"Loading pretrained model details for Project ID: "
                     f"{session_state.project.id} with deployment type: "
                     f"'{DEPLOYMENT_TYPE}'")
        if DEPLOYMENT_TYPE == "Semantic Segmentation with Polygons":
            # we need the `model_func` column
            for_display = False
        else:
            for_display = True
        models_df = get_pretrained_model_details(
            DEPLOYMENT_TYPE,
            for_display=for_display
        )

        # show description about the pretrained models, and also modify the df
        # to show only relevant models
        if DEPLOYMENT_TYPE == "Object Detection with Bounding Boxes":
            st.markdown(
                """This table is obtained from TensorFlow Object Detection Model Zoo 
                [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 
                The speed here actually means latency (ms), thus the lower the
                better; while higher COCO mAP score is better.""")
            txt_col, _ = st.columns([2, 1])
            with txt_col.expander("💡 Useful tips", expanded=True):
                st.markdown(
                    """- It is recommended to start with SSD MobileNet, then Faster RCNN 
models if performance is still not good enough. If there
is a significant improvement between SSD and Faster RCNN models,
then only you should consider trying EfficientNet or CenterNet.

- Note that higher COCO mAP score does not directly reflect its
performance on your dataset as it is highly dependent on the
type and quality of the dataset the model is being trained on.

- In general, it is always better for you to try to improve the
quality of your dataset rather than wasting time training with
different models, although training with higher number of steps
could improve its performance, you should always try smaller
number of steps to see whether it's performance on the test set
improves, otherwise it is not worth training any further.""")
            unwanted_idxs = models_df.loc[models_df['Model Name'].str.contains(
                "deprecated|Mask R-CNN|ExtremeNet")].index
            # drop deprecated models and Mask R-CNN model
            models_df = models_df.drop(unwanted_idxs).reset_index(drop=True)
        elif DEPLOYMENT_TYPE == "Image Classification":
            st.markdown(
                "This table is obtained from Keras available pretrained models for image classification "
                "[here](https://www.tensorflow.org/api_docs/python/tf/keras/applications).")
        else:
            # get only the models used for our training in this app
            model_func_names = list(
                get_segmentation_model_func2params().keys())
            models_df = models_df[models_df['model_func'].isin(
                model_func_names)]
            models_df = models_df.drop(
                columns='model_func').reset_index(drop=True)
            st.markdown(
                "This table is obtained from `keras-unet-collection`'s GitHub repository "
                "[here](https://github.com/yingkaisha/keras-unet-collection).")

        # get the index of the model in the table, either from previously submitted info,
        #  or from recommended model
        if training.attached_model is not None \
                and not training.attached_model.is_not_pretrained:
            # then this has already selected a model before
            model_name = training.attached_model.name
            model_idx = int(models_df.query(
                '`Model Name` == @model_name').index[0])
        else:
            if DEPLOYMENT_TYPE == "Object Detection with Bounding Boxes":
                # default to a good default SSD model to show to the user
                model_idx = 22
            elif DEPLOYMENT_TYPE == "Image Classification":
                model_idx = int(models_df.loc[
                    models_df['Model Name'] == "ResNet50"].index[0])
            else:
                model_idx = int(models_df.loc[
                    models_df['Model Name'] == 'U-net'].index[0])

        st.dataframe(models_df, width=1500)

        logger.debug(f"{model_idx = }")
        selected_model_name = st.selectbox(
            "Please select a Pre-trained model",
            options=models_df['Model Name'],
            index=model_idx, key="selected_pretrained_model",
        )
    if training.has_submitted[NewTrainingPagination.Model]:
        st.warning("WARNING: Changing the selected model will remove all existing "
                   "trained model data.")

    # Instantiate an attached model in the page
    model_df_row = Model.filtered_models_dataframe(
        models=existing_models,
        dataframe_col="Name",
        # filter_value=session_state.existing_models_table[0],
        # filter_value=session_state.existing_models_table,
        filter_value=selected_model_name,
        column_names=existing_models_column_names)
    selected_model = Model(model_row=model_df_row[0])

    # ******************************* DATA TABLE *******************************

    # >>>> MAIN_COL2

    # if training.attached_model:
    # with main_col2.container():

    # - can consider adding the bottom line after done creating self.model_input_size
    # Model Input Size: {training.attached_model.model_input_size}
    model_information = f"""
    ## Selected Model Information:
    ### Name:
    {selected_model.name}
    """
    # Framework:
    # {selected_model.framework}
    st.info(model_information)
    # >>>> GET PERF METRICS of SELECTED MODEL
    y = selected_model.get_perf_metrics()
    if y:
        df_metrics = Model.create_perf_metrics_table(y)
        st.write(f"#### Metrics:")
        st.table(df_metrics)
    session_state.new_training_place["attached_model_selection"] = st.empty(
    )

    # CALLBACK: CHECK IF MODEL NAME EXISTS
    def check_if_name_exist(field_placeholder, conn):
        input_name = session_state.training_model_name.strip()
        lower_input_name = input_name.lower()
        context = {'column_name': 'name',
                   'value': lower_input_name}

        with field_placeholder['training_model_name'].container():
            if lower_input_name:
                if training.training_model.check_if_exists(context, conn):
                    # training.training_model.name = ''
                    st.error("Model name used. Please enter a new name")
                    # sleep(0.7)
                    # field_placeholder['training_model_name'].empty()
                    logger.error(
                        f"Training Model name used. Please enter a new name")
                    st.stop()
                else:
                    # save the user's input one but check with lowercase one
                    # training.training_model.name = input_name
                    logger.info("Training Model name fresh and ready to rumble: "
                                f"'{input_name}'")
            else:
                st.error("Please enter a model name!")
                st.stop()

    # ***************************** Training model name *****************************
    st.write(f"**Step 2: Enter the name for your new training model** ")

    st.text_input('Model Name', key="training_model_name",
                  value=training.training_model.name,
                  help="Enter the name of the model to be exported after training",
                  #   on_change=check_if_name_exist,
                  #   args=(session_state.new_training_place, conn,)
                  )
    session_state.new_training_place['training_model_name'] = st.empty(
    )
    # st.write(lower_input_name)

    # ************************* MODEL DESCRIPTION (OPTIONAL) *************************
    description = st.text_area(
        "Description (Optional)", key="training_model_desc",
        value=training.training_model.desc,
        help="Enter the description of the model")

    if description:
        training.training_model.desc = remove_newline_trailing_whitespace(
            description)

    # ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    # new_training_section_back_button_place, _, \
    #     new_training_section_next_button_place = st.columns([0.5, 3, 0.5])
    new_training_section_next_button_place = st.empty()

    # typing.NamedTuple type

    # *************************************CALLBACK: NEXT BUTTON *************************************
    def to_training_configuration_page():
        # check_if_name_exist(session_state.new_training_place, conn)
        input_name = session_state.training_model_name.strip()
        lower_input_name = input_name.lower()

        if training.has_submitted[NewTrainingPagination.Model] and \
                training.training_model.name.lower() == lower_input_name:
            # no need to check whether the name exists if the user is using the same
            # existing submitted name
            logger.info("Existing model name is not changed")
            name_key = None
        else:
            name_key = 'training_model_name'

        new_training_model_submission_dict = NewTrainingSubmissionHandlers(
            insert=training.training_model.create_new_project_model_pipeline,
            update=training.training_model.update_new_project_model_pipeline,
            # check the lowercase input name, but insert user input name later
            context={
                'training_model_name': lower_input_name,
                # 'attached_model_selection': session_state.existing_models_table
            },
            name_key=name_key
        )

        # run submission according to current page
        # NEXT page if all constraints met

        if not training.has_submitted[NewTrainingPagination.Model]:
            # >>>> IF IT IS A NEW SUBMISSION
            submission_func = new_training_model_submission_dict.insert
        else:
            submission_func = new_training_model_submission_dict.update

        if training.training_model.check_if_field_empty(
                context=new_training_model_submission_dict.context,
                field_placeholder=session_state.new_training_place,
                name_key=new_training_model_submission_dict.name_key
        ):

            # run create/update project (attached) model pipeline
            if submission_func(
                    attached_model=selected_model,
                    project_name=session_state.project.name,
                    training_name=training.name,
                    training_id=training.id,
                    # insert/update the user input one, not the lowercase one
                    new_model_name=input_name
            ):
                if training.attached_model is not None and \
                        selected_model.name != training.attached_model.name:
                    logger.info(
                        "A different pretrained model from the existing config "
                        "has been selected. Making relevant changes ...")
                    # must change training config if model is changed
                    go_to_train_config = True
                    # also remove the trained model directory if attached model is changed
                    model_path = training.get_paths()['models']
                    if model_path.exists():
                        logger.info(
                            "Removing existing trained model directory")
                        shutil.rmtree(model_path)
                    # and reset all training progress
                    if training.is_started:
                        logger.info("Resetting training progress")
                        training_progress = {'Epoch': 0}
                        training.reset_training_progress(
                            training_progress)
                else:
                    go_to_train_config = False

                # NOTE: UPDATE THESE ONLY AFTER user has clicked submit button
                # set the model selected in the page to be our attached_model
                training.attached_model = selected_model

                # run update training attached model
                if training.update_training_attached_model(
                        attached_model_id=training.attached_model.id,
                        training_model_id=training.training_model.id):
                    # set has_submitted =True
                    training.has_submitted[NewTrainingPagination.Model] = True
                    logger.info(
                        f"Successfully created new training model {training.training_model.id}")

                    # Go to TrainingConfig page to avoid issues in case the model has
                    # been changed, especially for segmentation models, which could have
                    # different parameters for different pretrained models
                    if go_to_train_config:
                        training.has_submitted[NewTrainingPagination.TrainingConfig] = False

                    for page, submitted in training.has_submitted.items():
                        if not submitted:
                            session_state.new_training_pagination = page
                            break
                    else:
                        # go to Training page if all forms have been submitted
                        session_state.new_training_pagination = NewTrainingPagination.Training
                    logger.debug('New Training Pagination: '
                                 f'{session_state.new_training_pagination = }')
        else:
            logger.error("Error with submission form. Either field is not complete, "
                         "or info already exists in database.")
            st.error("Error with submission form. Either field is not complete, "
                     "or info already exists in database.")

    # ***************** NEXT BUTTON **************************
    if training.attached_model and \
        "CenterNet MobileNetV2" in training.attached_model.name:
        st.warning(
            """Please change your model from CenterNet MobileNetV2 to 
        something else as it experiences weird issues currently.""")
        st.stop()
    with new_training_section_next_button_place:
        if st.button("Submit Model Info", key="models_page_next_button"):
            to_training_configuration_page()
            st.experimental_rerun()


def index(RELEASE=True):
    # ****************** TEST ******************************
    # For testing:
    # Instantiate Project Class and pass name as argument
    # Instiate New Training Class and set name and ID

    if not RELEASE:
        logger.debug("Navigator: At Models INDEX")

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
        project_id_tmp = 9
        # for Anson: 2 for TFOD, 17 for img classif, 18 for segmentation
        # uploaded pet segmentation: 20
        training_id_tmp = 17
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if "new_training_pagination" not in session_state:
            session_state.new_training_pagination = NewTrainingPagination.Model

        if 'new_training' not in session_state:

            session_state.new_training = NewTraining(training_id_tmp,
                                                     project=session_state.project)
            session_state.new_training.name = "My Tenth Training"
            session_state.new_training.deployment_type = "Object Detection with Bounding Boxes"

        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        new_training_progress_bar = st.progress(0)
        new_training_progress_bar.progress(2 / 4)
        # ****************************** HEADER **********************************************
        st.write(f"## **Training Section:**")
    # ****************** TEST END ******************************

    # ************************ MODEL PAGINATION *************************
    models_page = {
        # old page before this models_page
        # ModelsPagination.TrainingInfoDataset: new_training_infodataset.infodataset,
        # the current page function in this script
        ModelsPagination.ExistingModels: existing_models,
        ModelsPagination.ModelUpload: user_model_upload_page,

        # ModelsPagination.TrainingConfig: new_training_training_config.training_configuration,
        # ModelsPagination.AugmentationConfig: new_training_augmentation_config.augmentation_configuration,
        # ModelsPagination.Training: run_training_page.index,
    }

    # ********************** SESSION STATE ******************************
    if 'models_pagination' not in session_state:
        session_state.models_pagination = ModelsPagination.ExistingModels
    if "new_training_place" not in session_state:
        session_state.new_training_place = {}

    # ***************** BACK BUTTON **************************
    def to_training_infodataset_page():
        # session_state.models_pagination = ModelsPagination.TrainingInfoDataset
        # must update this pagination variable too to make things work properly
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    # with new_training_section_back_button_place:
    st.sidebar.button("Back to Modify Training Info", key="models_page_back_button",
                      on_click=to_training_infodataset_page)

    # >>>> RETURN TO ENTRY PAGE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    existing_models_back_button_place = st.sidebar.empty()

    if session_state.models_pagination != ModelsPagination.ExistingModels:
        def to_existing_model_page():
            session_state.models_pagination = ModelsPagination.ExistingModels
            # Reset all widget attributes in User Model Upload page
            NewModel.reset_model_upload_page()

        existing_models_back_button_place.button("Back to Existing Models Dashboard",
                                                 key="back_to_existing_models_page",
                                                 on_click=to_existing_model_page)
    else:
        existing_models_back_button_place.empty()

    logger.debug(
        f"Entering Models Page: {session_state.models_pagination}")
    # >>>> MAIN FUNCTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    models_page[session_state.models_pagination]()

    # st.write("vars(session_state.new_training)")
    # st.write(vars(session_state.new_training))
    # st.write("vars(session_state.new_training.training_model)")
    # st.write(vars(session_state.new_training.training_model))
    # st.write("vars(session_state.new_training.attached_model)")
    # st.write(vars(session_state.new_training.attached_model))


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        chdir_root()  # change to root directory
        # False for debugging
        index(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
