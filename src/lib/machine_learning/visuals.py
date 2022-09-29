import math
from itertools import zip_longest
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

import streamlit as st

from object_detection.utils import visualization_utils as viz_utils

from core.utils.log import logger


def str2float(val: str, float_format: str = '.5g'):
    try:
        param_val = f"{float(val):{float_format}}"
    except Exception as e:
        logger.debug(f"Skip converting `{val}`: {e}")
        param_val = val
    return param_val


def pretty_format_param(param_dict: Dict[str, Any], float_format: str = '.5g',
                        st_newlines: bool = True, bold_name: bool = True) -> str:
    """
    Format param_dict (or any Dictionary) to become a nice output to show on Streamlit.

    `float_format` is used for formatting floats.
    The formatting for significant digits `.5g` is based on [StackOverflow](https://stackoverflow.com/questions/25780022/how-to-make-python-format-floats-with-certain-amount-of-significant-digits).

    Set both `st_newlines` and `bold_name` to True for displaying with `st.info`.
    Set them to False to display as one line, especially useful for table/dataframe.
    """
    config_info = []
    for k, v in sorted(param_dict.items()):
        if "_" in k:
            param_name = ' '.join(k.split('_')).capitalize()
        else:
            param_name = k.capitalize()
        if isinstance(v, dict):
            if v.values() is None:
                continue
            config_info.append(f"#### {param_name}")
            for nested_name, nested_v in v.items():
                param_val = str2float(str(nested_v), float_format)
                if bold_name:
                    current_info = f'**{nested_name}**: {param_val}'
                else:
                    current_info = f'{nested_name}: {param_val}'
                config_info.append(current_info)
        else:
            if v is None:
                continue
            param_val = str2float(str(v), float_format)
            if bold_name:
                current_info = f'**{param_name}**: {param_val}'
            else:
                current_info = f'{param_name}: {param_val}'
            config_info.append(current_info)
    if st_newlines:
        # this is the newline string to work for Streamlit (need double space)
        separator = '  \n'
    else:
        separator = '; '
    config_info = separator.join(config_info)
    return config_info


def prettify_db_metrics(data_list: Union[List[NamedTuple], List[Dict]],
                        return_dict: bool = False,
                        **kwargs) -> Union[List[NamedTuple], List[Dict]]:
    """Prettify Metrics in Dict or Namedtuple queried from database (DB) for displaying.

    Args:
        data_list (Union[List[namedtuple], List[dict]]): Query results from DB
        return_dict (bool, optional): True if query results of type Dict. Defaults to False.
        kwargs: Optional arguments to pass to `pretty_format_param`

    Returns:
        List: List of Formatted Metrics query results
    """
    prettified_data = []
    for data in data_list:
        # convert datetime with TZ to (2021-07-30 12:12:12) format
        if return_dict:
            data["Metrics"] = pretty_format_param(
                data["Metrics"],
                **kwargs)
        else:
            prettified_metrics = pretty_format_param(
                data.Date_Time,
                **kwargs)
            data = data._replace(
                Metrics=prettified_metrics)
        prettified_data.append(data)
    return prettified_data


def pretty_st_metric(
        metrics: Dict[str, Any],
        prev_metrics: Dict[str, Any],
        float_format: str = '.5g'):
    # ! DEPRECATED, use PrettyMetricPrinter class
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        # show green color when loss is reduced;
        # red color when increased
        delta_color = 'inverse'
        # get the previous value before prettifying it
        prev_val = prev_metrics[name]
        # prettifying the metric name for display
        name = ' '.join(name.split('_')).capitalize()
        # calculate the difference with previous metric value
        delta = val - prev_val
        # formatting the float values for display
        val = f"{val:{float_format}}"
        if delta == 0:
            # don't show any indicator if there is no difference, or
            # if it's the initial training metrics
            delta = None
        else:
            delta = f"{delta:{float_format}}"
        col.metric(name, val, delta, delta_color=delta_color)


@dataclass(order=False, eq=False)
class PrettyMetricPrinter:
    """
    Wrapper class for pretty print using [st.metric function](https://docs.streamlit.io/en/stable/api.html#streamlit.metric).
    This class is created mainly to store the previous metrics, to facilitate the
    calculation of the difference between the current and previous metric values.

    Args:
        float_format (str | Dict[str, str], optional): the formatting used for floats.
            Can pass in either a `str` to use the same formatting for all metrics, or pass in a `Dict` for different formatting for each metric.
            Defaults to `.5g` for 5 significant figures.
        delta_color (str | Dict[str, str]], optional): Similar to `float_format`, can pass in `str` or `Dict`.
            Defaults to `inverse` when the metric name contains `loss`, else `normal`.
            Refer to Streamlit docs for the effects on colors.
    """
    float_format: Union[str, Dict[str, str]] = '.5g'
    delta_color: Union[str, Dict[str, str]] = None
    prev_metrics: Dict[str, float] = field(default=None, init=False)

    def __post_init__(self):
        # more loss names to check for inverse delta_color
        self._extra_lossnames: Set[str] = {
            'categorical_crossentropy', 'iou_seg',
            'val_categorical_crossentropy', 'val_iou_seg'}
        self._first: bool = True

    def write(self, metrics: Dict[str, float]):
        """
        Use this to directly print out the current metrics in a nicely formatted way in columns and st.metric.
        metrics (Dict[str, Any]): The dictionary containing the metrics such as loss or accuracy
        """
        if self._first:
            self._first = False
            if not self.delta_color:
                self.delta_color = {}
                for name in metrics:
                    if 'loss' in name.lower() or name in self._extra_lossnames:
                        self.delta_color[name] = 'inverse'
                    else:
                        self.delta_color[name] = 'normal'
            if isinstance(self.float_format, str):
                self.float_format = {
                    name: self.float_format for name in metrics}
            self.prev_metrics = metrics.copy()

        cols = st.columns(len(metrics))
        for col, (name, val) in zip(cols, metrics.items()):
            # get the current parameters for the metric before updating them
            # and before prettifying the metric name
            delta_color = self.delta_color[name]
            float_format = self.float_format[name]
            prev_val = self.prev_metrics[name]
            # prettifying the metric name for display
            name = ' '.join(name.split('_')).capitalize()
            # calculate the difference with previous metric value
            delta = val - prev_val
            # formatting the float values for display
            val = f"{val:{float_format}}"
            if delta == 0:
                # don't show any indicator if there is no difference, or
                # if it's the initial training metrics
                delta = None
            else:
                delta = f"{delta:{float_format}}"
            # using the st.metric function here
            col.metric(name, val, delta, delta_color=delta_color)

        # updating previous metrics before proceeding
        self.prev_metrics = metrics.copy()


def create_class_colors(
        class_names: List[str],
        as_array: bool = False) -> Union[Dict[str, Tuple[int, int, int]], np.ndarray]:
    """Randomly assign colors for different classes. 

    `class_names` should be obtained from the `Trainer.class_names` attribute
    for more efficient computations. For COCO segmentation, this should obtain from
    `get_coco_classes`.

    `as_array` is required for coloring mask images with `get_colored_mask_image`.
    """
    rng = np.random.default_rng(21)
    colors = rng.integers(0, 255, size=(len(class_names), 3),
                          dtype=np.uint8)
    class_colors = {}
    for name, color in zip(class_names, colors):
        # must convert the NumPy dtypes to Python ints
        color = [int(c) for c in color]
        class_colors[name] = tuple(color)
    if 'background' in class_names:
        # set background to black color
        # NOTE: this is always the 0th index based on `get_coco_classes`
        class_colors['background'] = (0, 0, 0)

    if as_array:
        class_colors = np.array(list(class_colors.values()), dtype=np.uint8)
    logger.debug(f"{class_colors = }")
    return class_colors


def draw_gt_bboxes(
    image_np: np.ndarray,
    box_coordinates: Sequence[Tuple[int, int, int, int]],
    class_names: Union[List[str], str] = None,
    color: Tuple[int, int, int] = (0, 150, 0),
    class_colors: Dict[str, Tuple[int, int, int]] = None,
    copy_image: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on the image and return the drawn image as a copy.

    Args:
        image_np (np.ndarray): the image to be drawn
        box_coordinates (Sequence[Tuple[int, int, int, int]]): bounding box coordinates
            in the order used by Pascal VOC format: (xmin, ymin, xmax, ymax)
        class_names (Union[List[str], str], optional): a single class name `str` for only single class,
            or a `list` of class names to use for each bounding box. Defaults to None.
        color (Tuple[int, int, int], optional): color to use for the bounding boxes in
            this image. Defaults to (0, 150, 0).
        class_colors (Dict[str, Tuple[int, int, int]], optional): Can be created with the
            `create_class_colors` function. If this is passed in,
            these colors are used instead of the `color` passed in. Defaults to None.

    Returns:
        np.ndarray: the image drawn with bounding boxes
    """
    if copy_image:
        image_np = image_np.copy()
    logger.debug(f"Total annotations for the image: {len(box_coordinates)}")
    logger.debug(f"{class_names = }")

    if class_names is None:
        class_names = []
    elif isinstance(class_names, str):
        # set the label to be the same for each box
        class_names = [class_names] * len(box_coordinates)
    elif len(class_names) == 1:
        class_names = class_names * len(box_coordinates)

    for (xmin, ymin, xmax, ymax), class_name in zip_longest(box_coordinates, class_names):
        if isinstance(xmin, float):
            xmin, ymin = int(xmin), int(ymin)
            xmax, ymax = int(xmax), int(ymax)
        if class_colors:
            color = class_colors[class_name]
        cv2.rectangle(
            image_np,
            (xmin, ymin),
            (xmax, ymax),
            color=color,
            thickness=2)
        # draw the class name if given
        if class_name:
            y = ymin - 36 if ymin - 36 > 0 else ymin
            ((label_width, label_height), _) = cv2.getTextSize(
                class_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, thickness=2
            )

            cv2.rectangle(
                image_np,
                (xmin - 1, y),
                (
                    int(xmin + label_width * 1.02),
                    int(y + label_height + label_height * 1),
                ),
                color=color,
                thickness=cv2.FILLED,
            )

            cv2.putText(
                image_np,
                class_name,
                (
                    int(xmin + label_width * 0.02),
                    int(y + label_height + label_height * 0.5),
                ),  # bottom left
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.75,
                color=(255, 255, 255),
                thickness=2,
            )
    return image_np


def draw_tfod_bboxes(
        detections: Dict[str, Any],
        image_np: np.ndarray,
        category_index: Dict[int, Any],
        min_score_thresh: float = 0.6,
        is_checkpoint: bool = False) -> np.ndarray:
    """Draw TFOD detected bounding boxes on the image. Note that this does
    not create a new image copy for the purpose of faster computation.

    `category_index` is loaded using `load_labelmap` method"""
    if is_checkpoint:
        # NOTE: Model loaded from TFOD Checkpoint seems to need this
        label_id_offset = 1
        detection_classes = detections['detection_classes'] + label_id_offset
    else:
        detection_classes = detections['detection_classes']
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detection_classes,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False
    )


def get_colored_mask_image(image: np.ndarray,
                           mask: np.ndarray,
                           class_colors: np.ndarray,
                           alpha: Optional[float] = 0.5,
                           ignore_background: Optional[bool] = False) -> np.ndarray:
    """Get a colored mask image based on the given `mask` and `class_colors` array.

    Note that `class_colors` must be a `np.ndarray` for this to work. Can get this from
    `create_class_colors` with `as_array=True`.

    `mask` has unique pixel values starting from 0 to num_classes; and each pixel value
    is associated with a specific class and class color. 

    `alpha` is to control the transparency of overlayed colored mask. Defaults to 0.5.

    `ignore_background` is used to ignore the black background color when overlaying to
    make the output prettier. Note that this will cost a considerable amount of FPS.
    Defaults to False.
    """
    # given the class ID map obtained from the mask, we can map each of
    # the class IDs to its corresponding color
    colored_mask = class_colors[mask.astype(np.uint8)]
    if ignore_background:
        # note that this will reduce quite some FPS
        colored_mask = np.where(colored_mask == [0, 0, 0], image, colored_mask)

    # perform a weighted combination of the input image with the colored_mask to
    # form an output visualization with different colors for each class
    # this is same as the equation below but faster
    output = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)
    # output = (((1 - alpha) * image) +
    #           (alpha * colored_mask)).astype("uint8")
    return output


@st.cache
def create_color_legend(class_colors: Dict[str, Tuple[int, int, int]],
                        bgr2rgb: bool = True,
                        ignore_background: bool = True,
                        show_index: bool = False) -> np.ndarray:
    # initialize the settings, NOTE: these values are found to have the best results
    col_height, text_col_width, color_col_width = 30, 150, 300
    if show_index:
        text_col_width += 25
    if ignore_background and 'background' in class_colors:
        class_colors = class_colors.copy()
        del class_colors['background']
    # initialize the legend visualization
    legend = np.full(((len(class_colors) * col_height) +
                      col_height, color_col_width, 3), 255, dtype=np.uint8)
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(class_colors.items()):
        if show_index:
            className = f"{className} ({i})"
        # draw the class name + color on the legend
        color = [int(c) for c in color]
        cv2.putText(
            legend,
            className,
            (5, (i * col_height) + (col_height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 100, 0),
            2,
        )
        cv2.rectangle(legend, (text_col_width, (i * col_height)),
                      (color_col_width, (i * col_height) + col_height),
                      tuple(color), -1)
    if bgr2rgb:
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
    return legend


def get_cosine_lr(curr_step, base_lr, max_lr, warmup_steps, total_steps):
    if curr_step == total_steps:
        return base_lr
    elif curr_step < warmup_steps:
        return (max_lr - base_lr) * curr_step / warmup_steps + base_lr
    else:
        return base_lr + (max_lr - base_lr) \
            * (1 + math.cos(math.pi * (curr_step - warmup_steps)
                            / (total_steps - warmup_steps))) / 2


def plot_cosine_lr_schedule(
        learning_rate_base: int, total_steps: int,
        warmup_learning_rate: int, warmup_steps: int):
    series = [
        get_cosine_lr(
            i, warmup_learning_rate, learning_rate_base,
            warmup_steps, total_steps)
        for i in range(total_steps)
    ]

    fig = plt.figure(figsize=(15, 8))
    sns.set_style("darkgrid")
    sns.set_context('talk')
    sns.lineplot(data=series)
    plt.title("Learning rate")
    st.pyplot(fig)
