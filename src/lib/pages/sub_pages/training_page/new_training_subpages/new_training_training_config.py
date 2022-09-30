""" 
Title: Training Parameters Configuration (New Training Configuration)
Date: 11/9/2021 
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
from typing import Any, Dict

from tensorflow import keras
from keras_unet_collection import models
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>
# SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
# LIB_PATH = SRC / "lib"
# if str(LIB_PATH) not in sys.path:
#     sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from machine_learning.visuals import plot_cosine_lr_schedule  # logger
from training.training_management import NewTrainingPagination, Training
from project.project_management import Project
from user.user_management import User
from machine_learning.utils import NASNET_IMAGENET_INPUT_SHAPES
from training.utils import get_segmentation_model_name2func, get_training_param_from_session_state


def training_configuration(RELEASE=True):
    logger.debug("At new_training_training_config.py")

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
        project_id_tmp = 9
        logger.debug(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            logger.debug("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        if 'new_training' not in session_state:
            # for Anson: 2 for TFOD, 17 for img classif, 18 for segmentation
            session_state.new_training = Training(17, session_state.project)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")

    if "is_changing_default_lr" not in session_state:
        session_state.is_changing_default_lr = False

    st.markdown(f"### Step 2: Select training configuration.")

    DEPLOYMENT_TYPE = session_state.project.deployment_type
    training: Training = session_state.new_training
    param_dict = training.training_param_dict
    if not param_dict:
        # to change NoneType to a Dict
        param_dict = {}
    # logger.debug(f"{param_dict = }")
    msg_place = {}

    train_config_col, details_col = st.columns([1.5, 2])

    def check_segmentation_model(training_param: Dict[str, Any] = None):
        if not training_param:
            training_param = get_training_param_from_session_state()
        model_param_dict = training.get_segmentation_model_params(
            training_param
        )
        with train_config_col:
            try:
                with st.spinner("Building model with the selected parameters ..."):
                    keras.backend.clear_session()
                    getattr(models, model_func)(**model_param_dict)
                # raise Exception("DUMMY ERROR")
            except Exception as e:
                st.error("""Error building the segmentation model with the selected 
                parameters, please try changing the parameters and try again before 
                submitting.""")
                logger.error(
                    f"Error building keras_unet_collection segmentation model: {e}")
                st.stop()
            else:
                st.success(f"""🎉 **{model_name}** Model was built 
                successfully with the selected parameters! 
                You may proceed to submit the training config.""")

    with train_config_col:
        def update_training_param():
            # set this so that it can be saved to the training param
            session_state.param_is_changing_default_lr = session_state.is_changing_default_lr
            training_param = get_training_param_from_session_state(delete=True)
            if DEPLOYMENT_TYPE == "Semantic Segmentation with Polygons" \
                    and not training.attached_model.is_not_pretrained:
                # continue only if the model is built successfully
                check_segmentation_model(training_param)
            # update the database and our Training instance
            training.update_training_param(training_param)
            logger.debug(f"{training_param = }")
            training.has_submitted[NewTrainingPagination.TrainingConfig] = True
            logger.info(
                "Successfully submitted the selected training parameters")

            if not training.has_submitted[NewTrainingPagination.AugmentationConfig]:
                session_state.new_training_pagination = NewTrainingPagination.AugmentationConfig
            else:
                # go to Training page if all forms have been submitted
                session_state.new_training_pagination = NewTrainingPagination.Training
            logger.debug('New Training Pagination: '
                         f'{session_state.new_training_pagination}')

        if DEPLOYMENT_TYPE != "Object Detection with Bounding Boxes":
            # NOTE: these params will also be used for Semantic Segmentation for Keras training
            # details_col.write(param_dict)
            learning_rate = param_dict.get('learning_rate', 1e-4)
            optimizer = param_dict.get('optimizer', "Adam")
            bs = 32 if DEPLOYMENT_TYPE == 'Image Classification' else 8
            batch_size = param_dict.get('batch_size', bs)
            num_epochs = param_dict.get('num_epochs', 10)
            # NOTE: not using fine_tune_all for now
            # fine_tune_all = param_dict.get('fine_tune_all', False)
            image_size = param_dict.get('image_size', 224)

            # NOTE: store them in key names starting exactly with `param_`
            #  to be able to extract them and send them over to the Trainer for training
            # e.g. param_batch_size -> batch_size at the Trainer later
            if DEPLOYMENT_TYPE == "Image Classification":
                inp_choices = (32, 64, 128, 224, 256, 331, 512)
                model_name = training.attached_model.name
                if model_name in NASNET_IMAGENET_INPUT_SHAPES:
                    required_img_size = NASNET_IMAGENET_INPUT_SHAPES[model_name][0]
                    st.info(
                        f"Your pretrained model architecture '**{model_name}**' "
                        f"requires the input size of **{required_img_size}** to make use "
                        "of pretrained **ImageNet** weights. You should choose this size "
                        "if you want to utilize pretrained weights.")
                    if not param_dict:
                        # default to this size if it's new submission
                        image_size = required_img_size
            else:
                inp_choices = (128, 224, 256, 512)
            image_size = st.select_slider(
                "Input image size", inp_choices,
                value=image_size, key="param_image_size",
                help="""Image size to resize our image width and height into, e.g. 224 will
                resize our image into size of 224 x 224. Larger image size could result in
                better performance but most of the time it will just make the training
                unnecessarily longer without significant improvement. Recommended to just
                go with **224**, which is the most common input image size."""
            )
            if DEPLOYMENT_TYPE == "Semantic Segmentation with Polygons":
                # this is required as the first parameter to the keras_unet_collection model
                session_state['param_input_size'] = (image_size, image_size, 3)

            lr_choices = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
                          1e-3, 5e-3, 1e-2, 5e-2, 0.1)
            st.select_slider(
                "Learning rate", lr_choices,
                value=learning_rate,
                # show scientific notation
                format_func=lambda x: f"{x:.0e}",
                # format='%.0e',
                key="param_learning_rate",
                help="""This controls how much we want to update our model parameters
                during each training step. If too low, the model will not even be able to learn,
                or at least train very very slow. If too high, the model parameters will explode
                and will not be able to learn too. Thus, this is a very important parameter to choose wisely.
                Recommeded to select **1e-4** as it is the middle ground. FYI: 1e-4 = 0.0001"""
            )

            optimizer_opts = ("Adadelta", "Adagrad", "Adam", "Adamax",
                              "Nadam", "RMSprop", "SGD")
            st.selectbox(
                "Optimizer", optimizer_opts,
                index=optimizer_opts.index(optimizer),
                key="param_optimizer",
                help="""An optimizer is responsible for updating our model parameters
                and minimize the loss function (or error function) during training.
                Recommeded to start with **Adam**."""
            )
            bs_choices = (1, 2, 4, 8, 16, 32, 64, 128)
            st.select_slider(
                "Batch size", bs_choices,
                value=batch_size,
                key="param_batch_size",
                help="""Update batch size based on the system's memory you have.
                Higher batch size will need a higher memory. Recommended to start
                with **32** (for image classification) or **8** (for image segmentation),
                64 could be fine depending on how large is the pretrained model,
                i.e. how many parameters does it have. Reduce if memory warning happens.
                You may choose to increase if you believe your GPU has a lot more
                VRAM (aka memory) left."""
            )
            st.number_input(
                "Number of epochs", min_value=3, max_value=10_000,
                value=num_epochs, step=1,
                key="param_num_epochs",
                help="""Number of epochs to train your model. One epoch will go through
                our entire dataset for exactly once. Recommended to start with **10**."""
            )
            if DEPLOYMENT_TYPE == "Image Classification":
                # NOTE: not using fine_tune_all for now
                # st.checkbox(
                #     "Fine-tune all layers", value=fine_tune_all,
                #     key="param_fine_tune_all",
                #     help="""In most cases, our custom dataset is much smaller than the original dataset
                #     used to train the pretrained model, therefore, it is preferred to freeze
                #     (their parameters are not affected by training) all the pretrained layers,
                #     and only train the remaining layers which we will append to accommodate our
                #     custom dataset. But in some cases where we want to completely fine-tune the
                #     pretrained parameters to fit our custom dataset, we will run the training
                #     for a second time with all the pretrained model layers unfrozen.
                #     This may or may not improve the performance, depending on how much our
                #     custom dataset differs from the original dataset used for pretraining.
                #     Recommended to **start with the normal way first**, i.e. only fine-tune
                #     the last few layers (leave this unchecked)."""
                # )
                # semantic segmentation submit button will be shown later
                st.button("Submit Config", key='btn_training_config_submit',
                          on_click=update_training_param)
        else:
            # ******************************** TFOD config ********************************
            batch_size = param_dict.get('batch_size', 4)
            num_train_steps = param_dict.get('num_train_steps', 5000)
            learning_rate_base = param_dict.get('learning_rate_base', 0.005)
            warmup_learning_rate = param_dict.get('warmup_learning_rate', 1e-4)
            warmup_steps = param_dict.get('warmup_steps', 1000)

            with st.form(key='training_config_form'):
                bs_choices = (1, 2, 4, 8, 16, 32, 64, 128)
                st.select_slider(
                    "Batch size", bs_choices,
                    value=batch_size,
                    key="param_batch_size",
                    help="""Update batch size based on the system's memory you have.
                    Higher batch size will need a higher memory. Recommended to start
                    with **4**. Reduce if memory warning happens. Beware that our object
                    detection models requires a lot of memory, so do not try to simply increase
                    the batch size if you are not sure whether you have enough GPU memory."""
                )
                num_train_steps = st.number_input(
                    "Number of training steps", min_value=100,
                    # NOTE: this max_value should be adjusted according to our server limit
                    max_value=100_000,
                    value=num_train_steps,
                    step=100, key='param_num_train_steps',
                    help="Recommended to train for at least **2000** steps. "
                    "Checkpoint is saved at every 100 steps."
                )
                msg_place['num_train_steps'] = st.empty()

                if session_state.is_changing_default_lr:
                    change_lr_btn_name = "Use the default learning rate schedule"
                else:
                    change_lr_btn_name = "Change default learning rate schedule"
                def change_lr_state_cb():
                    session_state.is_changing_default_lr = not session_state.is_changing_default_lr

                st.form_submit_button(
                    change_lr_btn_name, on_click=change_lr_state_cb)

                if session_state.is_changing_default_lr:
                    st.number_input(
                        "Base learning rate of the cosine annealing schedule",
                        1e-5, 1.0,
                        value=learning_rate_base,
                        format="%f",
                        key="param_learning_rate_base",
                        help="""Peak learning rate. Note that adjusting this value
                        has a very significant impact on the training progress.
                        Too high or too low value would result in the model unable
                        to learn and predict properly.
                        """
                    )
                    st.number_input(
                        "Warm up learning rate", 1e-5, 1.0,
                        value=warmup_learning_rate,
                        format="%f",
                        key="param_warmup_learning_rate",
                        help="""Initial warm up learning rate"""
                    )
                    st.number_input(
                        "Warm up steps", 10,
                        value=warmup_steps,
                        key="param_warmup_steps",
                        help="""Number of steps to warm up until reach the base
                        learning rate"""
                    )
                    msg_place['warmup_steps'] = st.empty()

                    submit_plot = st.form_submit_button(
                        "Plot learning rate schedule")
                submit = st.form_submit_button("Submit Config")

            with details_col:
                def validate_lr_params():
                    if session_state.param_warmup_steps >= session_state.param_num_train_steps:
                        msg_place['warmup_steps'].error(
                            """Warm up steps must be less than the total
                        number of training steps.""")
                        st.stop()
                if session_state.is_changing_default_lr and submit_plot:
                    validate_lr_params()
                    lr_params = {
                        "learning_rate_base": session_state.param_learning_rate_base,
                        "num_train_steps": session_state.param_num_train_steps,
                        "warmup_learning_rate": session_state.param_warmup_learning_rate,
                        "warmup_steps": session_state.param_warmup_steps
                    }
                    logger.debug(f"Learning rate parameters: {lr_params}")
                    plot_cosine_lr_schedule(*lr_params.values())
            if submit:
                if session_state.is_changing_default_lr:
                    validate_lr_params()
                if num_train_steps % 100 != 0:
                    msg_place['num_train_steps'].error(
                        """Number of training steps must be a multiple of 100 because
                        the checkpoint is saved at every 100 steps, and this is required
                        to be able to properly continue training from previous steps
                        when necessary.""")
                    st.stop()
                update_training_param()
                st.experimental_rerun()

        # ****************** Model parameters for keras_unet_collection models ******************
        # no need these params if the attached_model is not pretrained (i.e. is uploaded
        #  or is a trained project model)
        if DEPLOYMENT_TYPE == "Semantic Segmentation with Polygons" \
                and not training.attached_model.is_not_pretrained:
            # NOTE: refer to Notion for details about the model parameters
            # or refer to this Colab Notebook https://colab.research.google.com/drive/1PgI3Adcq_EixOrZm5kFsjabswxIw0c4p?usp=sharing
            filter_num = param_dict.get('filter_num', [32, 64, 128, 256])
            # note: there is also an `n_labels` parameter initialized directly below
            filter_size = filter_num[0]
            depth = len(filter_num)
            recur_num = param_dict.get('recur_num', 2)
            stack_num_up = param_dict.get('stack_num_up', 2)
            stack_num_down = param_dict.get('stack_num_down', 2)
            activation = param_dict.get('activation', 'ReLU')
            output_activation = param_dict.get('output_activation', 'Softmax')
            batch_norm = param_dict.get('batch_norm', True)
            pool = param_dict.get('pool', True)
            unpool = param_dict.get('unpool', True)
            aspp_num_down = param_dict.get('aspp_num_down', 256)
            aspp_num_up = param_dict.get('aspp_num_up', 128)
            use_hybrid_loss = param_dict.get('use_hybrid_loss', False)

            st.markdown("___")
            st.subheader("Segmentation model parameters")

            filter_size_choices = (16, 32, 64)
            filter_size = st.select_slider(
                "Filter size for first layer", filter_size_choices,
                value=filter_size, key='filter_size', help="`filter_num` parameter"
            )
            depth_choices = (4, 5, 6)
            depth = st.select_slider(
                "Depth", depth_choices, value=depth, key='depth',
                help="""`depth` parameter. Number of filters per down/up-sampling blocks.
                e.g. If selected first layer filter size 32 with depth of 4:
                `[32, 64, 128, 256]`. i.e. 1st layer has 32 filters; 2nd: 64; 
                3rd: 128; 4th: 256. This is the conventional pattern for the 
                increasing number of filters"""
            )
            # e.g. Depth 4: [32, 64, 128, 256]
            session_state['param_filter_num'] = [
                filter_size * (2 ** i) for i in range(depth)]

            num_classes = session_state.project.get_num_classes()
            session_state['param_n_labels'] = num_classes

            recur_num_choices = (1, 2, 3)
            if training.attached_model.name == 'R2U-Net':
                st.select_slider(
                    "Number of recurrent iterations", recur_num_choices, value=recur_num,
                    key='param_recur_num', help="`param_recur_num` parameter.  \n"
                    "Number of recurrent iterations per down- and upsampling level"
                )

            if training.attached_model.name == 'ResUnet-a':
                # recommended dilation_num by author
                session_state['param_dilation_num'] = [1, 3, 5, 31]
                aspp_num_choices = (64, 128, 256)
                st.select_slider(
                    "Number of filters in ASPP up layer", aspp_num_choices, value=aspp_num_up,
                    key='param_aspp_num_up',
                    help="""`aspp_num_up` parameter. Number of Atrous Spatial Pyramid 
                    Pooling (ASPP) layer  \nfilters after the last upsampling block."""
                )
                st.select_slider(
                    "Number of filters in ASPP down layer", aspp_num_choices, value=aspp_num_down,
                    key='param_aspp_num_down',
                    help="""`aspp_num_down` parameter. Number of ASPP layer  \nfilters 
                    after the last downsampling block."""
                )
            else:
                stack_num_choices = (1, 2, 3)
                st.select_slider(
                    "Number of upsampling Conv layers", stack_num_choices, value=stack_num_up,
                    key='param_stack_num_up',
                    help="""`stack_num_up` parameter. Number of convolutional  \nlayers
                    (after concatenation) per upsampling level/block"""
                )
                st.select_slider(
                    "Number of downsampling Conv layers", stack_num_choices, value=stack_num_down,
                    key='param_stack_num_down',
                    help="""`stack_num_down` parameter. Number of convolutional  \nlayers
                    per downsampling level/block"""
                )

            # NOTE: 'GELU' and 'Snake' are custom objects, they can be loaded from
            # get_segmentation_model_custom_objects() later
            activation_choices = ('ReLU', 'LeakyReLU',
                                  'PReLU', 'ELU', 'GELU', 'Snake')
            st.selectbox(
                "Activation function", activation_choices,
                index=activation_choices.index(activation),
                key='param_activation'
            )

            # NOTE: 'Snake' is a custom object
            output_act_choices = ('Sigmoid', 'Softmax', 'Linear', 'Snake')
            # convert back from model param to the format to display to user
            output_activation = 'Linear' if output_activation is None else output_activation
            output_activation = st.selectbox(
                "Output activation function", output_act_choices,
                index=output_act_choices.index(output_activation),
                key='output_activation'
            )
            # convert to the parameter that can be accepted by the segmentation model
            output_activation = None if output_activation == 'Linear' else output_activation
            session_state['param_output_activation'] = output_activation

            st.checkbox("Batch Normalization", value=batch_norm, key='param_batch_norm',
                        help="Whether to use batch normalization layers or not")

            pool_choices = (False, True, 'Average')
            pool = 'Average' if pool == 'ave' else pool
            pool = st.selectbox(
                "Pooling layer", pool_choices,
                index=pool_choices.index(pool), key='pool',
                help="`pool` parameter. False for strided Conv layers;  \n"
                "True for Max Pooling;'Average' for Average Pooling"
            )
            session_state['param_pool'] = 'ave' if pool == 'Average' else pool

            unpool_choices = (False, True, 'Nearest')
            unpool = 'Nearest' if unpool == 'nearest' else unpool
            unpool = st.selectbox(
                "Unpooling layer", unpool_choices,
                index=unpool_choices.index(unpool), key='unpool',
                help="`unpool` parameter. True for Upsampling2D with bilinear "
                "interpolation.  \n"
                "'Nearest' for Upsampling2D with nearest interpolation.  \n"
                "False for Conv2DTranspose + batch norm + activation. "
            )
            session_state['param_unpool'] = 'nearest' if unpool == 'Nearest' else unpool

            loss_choices = ('Hybrid Loss', 'Focal Tversky Loss')
            loss_func = 'Hybrid Loss' if use_hybrid_loss else 'Focal Tversky Loss'
            loss_func = st.radio(
                'Loss function', loss_choices, index=loss_choices.index(loss_func), key='loss_func',
                help="Hybrid Loss is a combination of Focal Tversky Loss and  \n"
                "Intersection over Union (IoU) Loss.  \n"
                "In general, focal Tversky loss is good enough.")
            session_state['param_use_hybrid_loss'] = (True if loss_func == 'Hybrid Loss'
                                                      else False)

            st.button("Test Build Model", key='btn_test_build_model',
                      help="""Test building a segmentation model to verify that 
                      the parameters are working""")

            with details_col:
                model_name2_func = get_segmentation_model_name2func()
                model_name = training.attached_model.name
                model_func = model_name2_func[model_name]
                # show docstring
                st.subheader(f"**{model_name}** Model Docstring:")
                st.text(getattr(models, model_func).__doc__)

                # using this instead of `on_click` callbacks to show the messages
                # below the other texts
                if session_state.btn_test_build_model:
                    check_segmentation_model()

    if DEPLOYMENT_TYPE == "Semantic Segmentation with Polygons":
        st.button("Submit Config", key='btn_training_config_submit',
                  on_click=update_training_param)
    # ******************************BACK BUTTON******************************

    def to_models_page():
        session_state.new_training_pagination = NewTrainingPagination.Model

    st.sidebar.button("Back to Modify Model Info", key="training_config_back_button",
                      on_click=to_models_page)


if __name__ == "__main__":
    # DEFINE wide page layout for debugging when running this page directly
    # layout = 'wide'
    # st.set_page_config(page_title="Integrated Vision Inspection System",
    #                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

    if st._is_running_with_streamlit:
        training_configuration(RELEASE=False)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
