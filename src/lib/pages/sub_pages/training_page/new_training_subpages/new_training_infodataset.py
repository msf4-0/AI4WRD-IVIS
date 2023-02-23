"""

Title: New Training InfoDataset
Date: 3/9/2021
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
from time import sleep
from typing import Union

import streamlit as st

# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state
from training.model_management import ModelsPagination
from training.training_management import (
    NewTraining,
    NewTrainingPagination,
    NewTrainingSubmissionHandlers,
    Training,
)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# DEFINE Web APP page configuration
# layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# <<<< Variable Declaration <<<<

chdir_root()


def infodataset():
    logger.debug(
        "[NAVIGATOR] At `new_training_infodataset.py` `infodataset` function")
    if 'new_training_place' not in session_state:
        session_state.new_training_place = {}
    if 'new_training_pagination' not in session_state:
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset

    training: Union[NewTraining, Training] = session_state.new_training
    # ************COLUMN PLACEHOLDERS *****************************************************
    st.write("___")

    # to display existing Training info for the users
    existing_info_place = st.empty()

    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    info_dataset_divider = st.empty()

    # create 2 columns for "New Data Button"
    datasetcol1, datasetcol2, datasetcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Dataset Dataframe buttons
    _, dataset_button_col1, _, dataset_button_col2, _, dataset_button_col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> New Training INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if training.has_submitted[NewTrainingPagination.InfoDataset]:
        # display existing information to the users for easier reference when updating
        # existing training info
        with existing_info_place.container():
            st.info(f"""
            **Current Training Title**: {training.name}  \n
            **Current Description**: {training.desc}  \n
            **Current Dataset List**: {training.dataset_chosen}  \n
            **Current Partition Ratio**: {training.partition_ratio}  \n
            """)

    infocol1.write("## __Training Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        input_name = session_state.new_training_name.strip()
        lower_input_name = input_name.lower()

        context = {'column_name': 'name',
                   'value': lower_input_name}

        logger.debug(f"New Training: {context}")

        if lower_input_name:
            if training.name.lower() == lower_input_name:
                logger.debug("Not changing name")
                return

            if NewTraining.check_if_exists(context, conn):

                # training.name = ''
                field_placeholder['new_training_name'].error(
                    f"Training name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_training_name'].empty()
                logger.error(f"Training name used. Please enter a new name")

            # else:
            #     training.name = input_name
            #     logger.info(f"Training name fresh and ready to rumble")

    with infocol2.container():

        # **** TRAINING TITLE ****
        st.text_input(
            "Training Title", key="new_training_name",
            value=training.name,
            help="Enter the name of the training",
            on_change=check_if_name_exist, args=(session_state.new_training_place, conn,))
        session_state.new_training_place["new_training_name"] = st.empty()

        # **** TRAINING DESCRIPTION (Optional) ****
        description = st.text_area(
            "Description (Optional)", key="new_training_desc",
            value=training.desc,
            help="Enter the description of the training")

        if description:
            description = remove_newline_trailing_whitespace(description)

    # <<<<<<<< New Training INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    info_dataset_divider.write("___")

    datasetcol1.write("## __Dataset :__")

    # ******************************* Right Column to select dataset *******************************
    with datasetcol3.container():

        # >>>> Store SELECTED DATASET >>>>
        dataset_chosen = st.multiselect(
            "Dataset List", key="new_training_dataset_chosen",
            default=training.dataset_chosen,
            options=session_state.project.dataset_dict, help="Assign dataset to the training")
        session_state.new_training_place["new_training_dataset_chosen"] = st.empty(
        )

        if not dataset_chosen:
            st.info("Please select at least a dataset.")
            st.stop()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATASET PARTITION CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def update_partition_ratio():
            if round(session_state.input_train_ratio + session_state.input_eval_ratio
                     + session_state.input_test_ratio, 2) != 1.00:
                partition_ratio_error_place.error(
                    'Dataset ratios must sum up to 1.0!')
                st.stop()

            session_state.partition_ratio['train'] = session_state.input_train_ratio
            session_state.partition_ratio['eval'] = session_state.input_eval_ratio
            session_state.partition_ratio['test'] = session_state.input_test_ratio

        if 'partition_ratio' not in session_state:
            partition_ratio = training.partition_ratio.copy()
            if round(partition_ratio['test'], 2) == 0.00:
                # For background compatible!! To avoid issues when test set
                # was set to 0 previously
                partition_ratio['test'] = partition_ratio['eval']
                partition_ratio['eval'] = 0.00
                training.partition_ratio = partition_ratio.copy()
                # update the database for the correct partition_ratio
                training.update_training_info()
            session_state.partition_ratio = partition_ratio

        partition_ratio = session_state.partition_ratio

        st.markdown('Dataset Partition Ratio')
        st.number_input(
            'Training set ratio', 0.5, 0.99, value=partition_ratio['train'],
            step=0.01, format='%.2f', key='input_train_ratio')
        st.number_input(
            'Validation set ratio (optional)', 0., 0.99, value=partition_ratio['eval'],
            step=0.01, format='%.2f', key='input_eval_ratio',
            help='Optional but recommended. Useful for an extra validation set '
            'used during training for model evaluation purposes before '
            'training has finished.')
        st.number_input(
            'Testing set ratio', 0.01, 0.99, value=partition_ratio['test'],
            step=0.01, format='%.2f', key='input_test_ratio')
        partition_ratio_error_place = st.empty()

        update_partition_ratio()

        partition_size = training.calc_dataset_partition_size(
            partition_ratio, dataset_chosen,
            session_state.project.dataset_dict)

        st.info(f"""
        #### Train Dataset Ratio: {partition_ratio['train']} ({partition_size['train']} data)
        #### Evaluation Dataset Ratio: {partition_ratio['eval']} ({partition_size['eval']} data)
        #### Test Dataset Ratio: {partition_ratio['test']} ({partition_size['test']} data)
        """)

        # >>>> DISPLAY DATASET CHOSEN >>>>
        st.write("### Dataset chosen:")
        for idx, data in enumerate(dataset_chosen):
            st.write(f"{idx+1}. {data}")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATASET PARTITION CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ******************************* Right Column to select dataset *******************************

    # ******************* Left Column to show full list of dataset and selection *******************
    if "dataset_page" not in session_state:
        session_state.new_training_dataset_page = 0

    with datasetcol2.container():
        start = 10 * session_state.new_training_dataset_page
        end = start + 10

        # >>>>>>>>>>PANDAS DATAFRAME >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        df = create_dataframe(session_state.project.datasets,
                              column_names=session_state.project.column_names,
                              sort=True, sort_by='ID', asc=True, date_time_format=True)

        df_loc = df.loc[:, "ID":"Date/Time"]
        df_slice = df_loc.iloc[start:end]

        # GET color from active theme
        df_row_highlight_color = get_df_row_highlight_color()

        def highlight_row(x, selections):

            if x.Name in selections:

                return [f'background-color: {df_row_highlight_color}'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=dataset_chosen, axis=1)

        # >>>>DATAFRAME
        # st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
        #     [dict(selector='th', props=[('text-align', 'center')])]))
        st.dataframe(styler, height=800)
    # ******************* Left Column to show full list of dataset and selection *******************

    # **************************************** DATASET PAGINATION ****************************************

    # >>>> PAGINATION CALLBACK >>>>
    def next_page():
        session_state.new_training_dataset_page += 1

    def prev_page():
        session_state.new_training_dataset_page -= 1

    # _, col1, _, col2, _, col3, _ = st.columns(
    #     [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    num_dataset_per_page = 10
    num_dataset_page = len(
        session_state.project.dataset_dict) // num_dataset_per_page

    if num_dataset_page > 1:
        if session_state.new_training_dataset_page < num_dataset_page:
            dataset_button_col3.button(">", on_click=next_page)
        else:
            # this makes the empty column show up on mobile
            dataset_button_col3.write("")

        if session_state.new_training_dataset_page > 0:
            dataset_button_col1.button("<", on_click=prev_page)
        else:
            # this makes the empty column show up on mobile
            dataset_button_col1.write("")

        dataset_button_col2.write(
            f"Page {1+session_state.new_training_dataset_page} of {num_dataset_page}")
    # **************************************** DATASET PAGINATION ****************************************
# <<<<<<<< Choose Dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    _, _, new_training_section_next_button_place = st.columns([1, 3, 1])

    # >>>> NEXT BUTTON >>>>
    def to_new_training_next_page():
        # Run submission according to current page
        # NEXT page if constraints are met

        if isinstance(training, Training):
            # Training instance will not need to insert new info anymore, just need to update
            def insert_function():
                return None
        else:
            insert_function = training.insert_training_info_dataset

        input_name = session_state.new_training_name.strip()
        lower_input_name = input_name.lower()

        if training.name.lower() == lower_input_name:
            # no need to check whether the name exists if the user is using the same
            # existing submitted name
            logger.info("Existing training name is not changed")
            name_key = None
        else:
            name_key = 'new_training_name'

        # typing.NamedTuple type
        new_training_infodataset_submission_dict = NewTrainingSubmissionHandlers(
            insert=insert_function,
            update=training.update_training_info_dataset,
            context={
                # check the lowercase input name, but insert user input name later
                'new_training_name': lower_input_name,
                'new_training_dataset_chosen': dataset_chosen
            },
            name_key=name_key
        )

        if not NewTraining.check_if_field_empty(
                context=new_training_infodataset_submission_dict.context,
                field_placeholder=session_state.new_training_place,
                name_key=new_training_infodataset_submission_dict.name_key):
            st.stop()

        # >>>> IF IT IS A NEW SUBMISSION
        if not training.has_submitted[NewTrainingPagination.InfoDataset]:
            # INSERT Database
            # Training Name,Desc, Dataset chosen, Partition Size
            training.dataset_chosen = dataset_chosen
            if new_training_infodataset_submission_dict.insert(
                    # insert the user input one, not the lowercase one
                    input_name,
                    description, partition_ratio):
                session_state.new_training_pagination = NewTrainingPagination.Model
                # must set this to tell the models_page.py to move to stay in its page
                session_state.models_pagination = ModelsPagination.ExistingModels
                training.has_submitted[NewTrainingPagination.InfoDataset] = True
                logger.info(
                    f"Successfully created new training {training.id}")

        # >>>> UPDATE if Training has already been submitted prior to this
        elif training.has_submitted[NewTrainingPagination.InfoDataset]:
            # UPDATE Database
            # Training Name,Desc, Dataset chosen, Partition Size
            if new_training_infodataset_submission_dict.update(
                    partition_ratio, dataset_chosen,
                    session_state.project.dataset_dict,
                    # update with the user input one, not the lowercase one
                    name=input_name,
                    desc=description):
                # session_state.new_training_pagination = NewTrainingPagination.Model
                # # must set this to tell the models_page.py to move to stay in its page
                # session_state.models_pagination = ModelsPagination.ExistingModels

                for page, submitted in training.has_submitted.items():
                    if not submitted:
                        session_state.new_training_pagination = page
                        break
                else:
                    # go to Training page if all forms have been submitted
                    session_state.new_training_pagination = NewTrainingPagination.Training

                logger.info(
                    f"Successfully updated new training {training.id}")
                logger.debug('New Training Pagination: '
                             f'{session_state.new_training_pagination}')

    with new_training_section_next_button_place:
        if st.button("Next", key="new_training_next_button"):
            to_new_training_next_page()
            st.experimental_rerun()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        infodataset()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
