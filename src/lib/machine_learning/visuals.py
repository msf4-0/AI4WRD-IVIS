from itertools import zip_longest
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import cv2

import streamlit as st

from object_detection.utils import visualization_utils as viz_utils

from core.utils.log import logger


def pretty_format_param(param_dict: Dict[str, Any], float_format: str = '.5g') -> str:
    """
    Format param_dict to become a nice output to show on Streamlit.
    `float_format` is used for formatting floats.
    The formatting for significant digits `.5g` is based on [StackOverflow](https://stackoverflow.com/questions/25780022/how-to-make-python-format-floats-with-certain-amount-of-significant-digits).
    """
    def str2float(val: str):
        try:
            param_val = f"{float(val):{float_format}}"
        except Exception as e:
            logger.debug(f"Skip converting `{val}``: {e}")
            param_val = val
        return param_val

    config_info = []
    for k, v in param_dict.items():
        if "_" in k:
            param_name = ' '.join(k.split('_')).capitalize()
        else:
            param_name = k.capitalize()
        if isinstance(v, dict):
            if not v.values():
                continue
            config_info.append(f"#### {param_name}")
            for nested_name, nested_v in v.items():
                param_val = str2float(nested_v)
                current_info = f'**{nested_name}**: {param_val}'
                config_info.append(current_info)
        else:
            if not v:
                continue
            param_val = str2float(v)
            current_info = f'**{param_name}**: {param_val}'
            config_info.append(current_info)
    config_info = '  \n'.join(config_info)
    return config_info


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

    def write(self, metrics: Dict[str, float]):
        """
        Use this to directly print out the current metrics in a nicely formatted way in columns and st.metric.
        metrics (Dict[str, Any]): The dictionary containing the metrics such as loss or accuracy
        """
        if not self.delta_color:
            self.delta_color = {
                name: 'inverse'
                if 'loss' in name
                else 'normal'
                for name in metrics
            }
        if isinstance(self.float_format, str):
            self.float_format = {name: self.float_format for name in metrics}
        if not self.prev_metrics:
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


@st.experimental_memo
def create_class_colors(
        class_names: List[str],
        as_array: bool = False) -> Union[Dict[str, Tuple[int, int, int]], np.ndarray]:
    """Randomly assign colors for different classes. 

    `class_names` should be obtained from the `Trainer.class_names` attribute
    for more efficient computations.

    `as_array` is required for coloring mask images with `get_colored_mask_image`.
    """
    np.random.seed(42)
    colors = np.random.randint(0, 255,
                               size=(len(class_names), 3),
                               dtype=np.uint8)
    class_colors = {}
    if 'background' in class_names:
        # set background to black color
        # NOTE: this should be the 0th index for segmentation so this must come first
        class_colors['background'] = (0, 0, 0)
    for name, color in zip(class_names, colors):
        color = [int(c) for c in color]
        class_colors[name] = tuple(color)

    if as_array:
        class_colors = np.array(list(class_colors.values()))
    return class_colors


def draw_gt_bbox(
    image_np: np.ndarray,
    box_coordinates: Sequence[Tuple[int, int, int, int]],
    class_names: Union[List[str], str] = None,
    color: Tuple[int, int, int] = (0, 150, 0),
    class_colors: Dict[str, Tuple[int, int, int]] = None
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
    image_with_gt_box = image_np.copy()
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
            image_with_gt_box,
            (xmin, ymin),
            (xmax, ymax),
            color=color,
            thickness=2)
        # draw the class name if given
        if class_name:
            ((label_width, label_height), _) = cv2.getTextSize(
                class_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, thickness=2
            )

            cv2.rectangle(
                image_with_gt_box,
                (xmin, ymin),
                (
                    int(xmin + label_width + label_width * 0.05),
                    int(ymin + label_height + label_height * 1),
                ),
                color=color,
                thickness=cv2.FILLED,
            )

            cv2.putText(
                image_with_gt_box,
                class_name,
                (
                    int(xmin + label_width * 0.03),
                    int(ymin + label_height + label_height * 0.5),
                ),  # bottom left
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.75,
                color=(255, 255, 255),
                thickness=2,
            )
    return image_with_gt_box


def draw_tfod_bboxes(
        detections: Dict[str, Any],
        image_np: np.ndarray,
        category_index: Dict[int, Any],
        min_score_thresh: float = 0.6) -> np.ndarray:
    """`category_index` is loaded using `load_labelmap` method"""
    label_id_offset = 1  # might need this
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        # detections['detection_classes'] + label_id_offset,
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False
    )
    return image_np_with_detections


def get_colored_mask_image(image: np.ndarray,
                           mask: np.ndarray,
                           class_colors: np.ndarray,
                           image_weight: Optional[float] = 0.4,
                           mask_weight: Optional[float] = 0.6) -> np.ndarray:
    """Get a colored mask image based on the given `mask` and `class_colors` array.

    Note that `class_colors` must be a `np.ndarray` for this to work. Can get this from
    `create_class_colors` with `as_array=True`.

    `mask` has unique pixel values starting from 0 to num_classes; and each pixel value
    is associated with a specific class and class color. 

    `image_weight` and `mask_weight` is to control
    their transparency of overlay.
    """
    # given the class ID map obtained from the mask, we can map each of
    # the class IDs to its corresponding color
    colored_mask = class_colors[mask.astype(np.uint8)]
    colored_mask = cv2.resize(
        colored_mask, (image.shape[1], image.shape[0]
                       ), interpolation=cv2.INTER_NEAREST
    )
    # perform a weighted combination of the input image with the colored_mask to
    # form an output visualization with different colors for each class
    output = ((image_weight * image) +
              (mask_weight * colored_mask)).astype("uint8")
    return output
