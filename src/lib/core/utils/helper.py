"""
Title: Helper
Date: 19/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

from datetime import datetime
import logging
import mimetypes
import sys
from enum import IntEnum
from functools import wraps
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import re
import numpy as np
# from imutils.video.webcamvideostream import WebcamVideoStream
from core.webcam.webcamvideostream import WebcamVideoStream

import pytz
import cv2
import pandas as pd
from psutil import virtual_memory, cpu_percent
import streamlit as st
from streamlit import session_state

from colorutils import hex_to_hsv
from streamlit.uploaded_file_manager import UploadedFile

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass
from core.color_extract import color_extract
from core.utils.log import logger  # logger
from core.utils.form_manager import remove_newline_trailing_whitespace
from data_manager.database_manager import db_fetchone, init_connection
# >>>> User-defined Modules >>>>
from path_desc import chdir_root

conn = init_connection(**st.secrets["postgres"])


DATETIME_STR_FORMAT = "%Y-%m-%d_%H-%M-%S_%f"


@st.experimental_memo
def get_all_timezones() -> Tuple[str, ...]:
    return tuple(pytz.all_timezones)


def get_now_string(dt_format=DATETIME_STR_FORMAT, timezone="Singapore") -> str:
    local_tz = pytz.timezone(timezone)
    tz_now = datetime.now(tz=local_tz)
    return tz_now.strftime(dt_format)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, description: str = '', disable: bool = None,
                 logging_level: int = logging.DEBUG):
        self.description = description
        self._start_time = None
        if disable is not None:
            # optionally enable/disable it ONLY when using context manager
            self._disabled = disable
        else:
            # defaults to False
            self._disabled = False

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        time_elapsed = perf_counter() - self._start_time
        self._start_time = None
        logger.info(f"{self.description} [{time_elapsed:.4f} seconds]")

    def __enter__(self):
        """Start a new timer as a context manager"""
        if not self._disabled:
            self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        if not self._disabled:
            self.stop()


def get_ram_usage():
    """Return the current RAM usage in percentage."""
    return virtual_memory().percent


def get_cpu_usage(interval: float = 4):
    """Get CPU utilization over a specified interval.

    NOTE: this function is a blocking operation, e.g. interval=4 will
    block for 4 seconds to calculate CPU usage over time"""
    return cpu_percent(interval)


class HSV(NamedTuple):
    H: int
    S: int
    V: int


chdir_root()


def hex_to_hsv_converter(hex_code):
    hsv = HSV._make(hex_to_hsv(hex_code))

    return hsv


def get_df_row_highlight_color(color=None):
    color = color_extract(key='color_ts')
    color = color['backgroundColor'] if color else '#FFFFFF'

    value_threshold = 0.5
    dark_green = "#80CBC4"
    light_green = "#00796B"

    hsv = hex_to_hsv_converter(color)

    V = hsv.V

    df_row_highlight_color = dark_green if (
        V > value_threshold) else light_green

    return df_row_highlight_color


def get_textColor():
    text_color = (color_extract(key='text_color'))['textColor']
    return text_color


class NavColor(NamedTuple):
    border: str
    background: str


current_page = NavColor('#0071BC', '#29B6F6')
non_current_page = NavColor('#0071BC', None)
color = {'current_page': current_page, 'non_current_page': non_current_page}


def get_current_page_nav_color(index: int, num_pages: int, offset: int = 0) -> List[NavColor]:
    color = []

    for i in range(num_pages):
        color[i] = current_page if (
            i == (index - offset)) else non_current_page

    return color


def get_theme():
    backgroundColor = st.get_option('theme.backgroundColor')
    return backgroundColor


def remove_suffix(filename: Union[str, Path]) -> str:
    """Remove suffix/suffixes from a string

    Args:
        filename (Union[str, Path]): Filename or path-like object

    Returns:
        str: Formatted string without suffixes
    """
    suffix_removed = str(filename).replace(
        ''.join(Path(filename).suffixes), '')

    return suffix_removed


def split_string(string: str, separator: str = ' ') -> List:

    # Split the string based on space delimiter
    list_string = string.split(separator)

    return list_string


def join_string(list_string: List, separator: str = '-') -> str:

    # Join the string based on '-' delimiter
    string = separator.join(list_string)

    return string


def get_directory_name(name: str) -> str:
    """Get the proper directory name for dataset/model/training

    e.g. ' Dummy dataset 1 ' -> 'dummy-dataset-1'
    """
    # NOTE: this function is directly related to the Project.query_annotations()
    directory_name = join_string(split_string(
        remove_newline_trailing_whitespace(str(name)))).lower()
    # replace symbols that are unwanted in file/folder names
    directory_name = re.sub(r'[?/\\*"><|\s]+', '-', directory_name)
    return directory_name


def is_empty(iterable: Union[List, Dict, set]) -> bool:
    return not bool(iterable)


# @st.cache
def create_dataframe(data: Union[List, Dict, pd.Series],
                     column_names: List = None,
                     sort: bool = False,
                     sort_by: Optional[str] = None,
                     asc: bool = True,
                     date_time_format: bool = False) -> pd.DataFrame:
    if data:
        df = pd.DataFrame(data, columns=column_names)
        df.index.name = 'No.'
        if date_time_format:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'],
                                             format='%Y-%m-%d %H:%M:%S')

            # df.sort_values(by=['Date/Time'], inplace=True,
            #                ascending=False, ignore_index=True)
        if sort:
            df.sort_values(by=[sort_by], inplace=True,
                           ascending=asc, ignore_index=True)

            # dfStyler = df.style.set_properties(**{'text-align': 'center'})
            # dfStyler.set_table_styles(
            #     [dict(selector='th', props=[('text-align', 'center')])])

        return df


def dataframe2dict(orient='index') -> List[Dict[str, Any]]:

    def inner(func):
        @wraps(func)
        def convert_to_dict(*args, **kwargs) -> List[dict]:

            if args:
                df = func(*args)

            elif kwargs:
                df = func(**kwargs)

            dataframe_dict = list(df.to_dict(orient=orient).values())

            return dataframe_dict
        return convert_to_dict
    return inner


def datetime_formatter(data_list: Union[List[NamedTuple], List[Dict]], return_dict: bool = False) -> List:
    """Convert datetime format to %Y-%m-%d %H:%M:%S for Dict and namedtuple from DB query
    NOTE: CAN JUST use to_char(<column_name>, 'YYYY-MM-DD HH24:MI:SS') in query instead!


    Args:
        data_list (Union[List[namedtuple], List[dict]]): Query results from DB
        return_dict (bool, optional): True if query results of type Dict. Defaults to False.

    Returns:
        List: List of Formatted Date/Time query results
    """
    data_tmp = []
    for data in data_list:
        # convert datetime with TZ to (2021-07-30 12:12:12) format
        if return_dict:
            converted_datetime = data["Date/Time"].strftime(
                '%Y-%m-%d %H:%M:%S')
            data["Date/Time"] = converted_datetime
        else:
            converted_datetime = data.Date_Time.strftime(
                '%Y-%m-%d %H:%M:%S')

            data = data._replace(
                Date_Time=converted_datetime)
        data_tmp.append(data)

    return data_tmp


def get_dataframe_row(row_id: int, df: pd.DataFrame) -> Dict:
    """Get Data row from DataFrame

    Args:
        row_id (int): Row ID of DataFrame
        df (pd.DataFrame): Pandas DataFrame of interest

    Raises:
        TypeError: Only type `int` supported for row_id

    Returns:
        Dict: Data row of type Dict
    """

    # Handle data_id exceptions
    if isinstance(row_id, int):
        pass
    elif isinstance(row_id, Union[List, tuple]):
        assert len(row_id) <= 1, "Data selection should be singular"
        row_id = row_id[0]
    else:
        raise TypeError("Data ID can only be int")

    df_row = ((df.loc[df["id"] == row_id]
               ).to_dict(orient='records'))[0]

    return df_row


def get_identifier_str_IntEnum(identifier: Union[str, IntEnum],
                               enumerator_class: IntEnum, identifier_dictionary: Dict,
                               string: bool = False):

    if string:
        # Get String form if is type IntEnum class
        if isinstance(identifier, enumerator_class):
            identifier = [
                k for k, v in identifier_dictionary.items() if v == identifier][0]
    else:
        # Get IntEnum class constant if is string
        if isinstance(identifier, str):
            identifier = identifier_dictionary.get(identifier)

    logger.debug(f"Type is: {identifier!r}")

    return identifier


def get_mime(file: Union[str, Path]):
    """Get MIME type of file

    Args:
        file (Union[str, Path]): filepath in string or path-like object

    Returns:
        str: MIME type of file
    """
    mime = mimetypes.guess_type(file)[0]
    return mime


# MIME: type/subtype
# get filetype
def get_filetype(file: Union[str, Path, UploadedFile]):
    """Get filetype from MIME of the file <type/subtype>
        Eg. image,video,audio,text

    Args:
        file (Union[str,Path,UploadedFile]): File can be string path or Path-like object or Streamlit's UploadedFile object

    Returns:
        string: filetype
    """
    if isinstance(file, (str, Path)):
        mime_type = mimetypes.guess_type(file)[0]
        # filetype = str(Path(mime_type).parent)
    elif isinstance(file, UploadedFile):
        logger.info(f"File: {file}")
        mime_type = file.type
        file.seek(0)

    if mime_type:
        filetype = str(Path(mime_type).parent)
    else:
        filetype = None

    return filetype


def compare_filetypes(file_tuple: tuple):
    """Compare if 2 instances are equal, else break loop

    Args:
        file_tuple (tuple): A pair of elements to be compared -> 2 elements zip of a List or Dict

    Returns:
        bool: True is if equal, else False
    """
    filetype1, filetype2 = list(map(get_filetype, file_tuple))  # updated
    logger.info(f"File tuple:{file_tuple}")
    if filetype1 == filetype2:
        logger.info(f"File types:{filetype1,filetype2}")
        return True, filetype1, file_tuple[0].name, file_tuple[1].name
    else:

        return False, filetype1, file_tuple[0].name, file_tuple[1].name


def check_filetype(uploaded_files, dataset, field_placeholder: Dict = None):
    """Constraint for only one type of files (Image, Video, Audio, Text)

    1. Image: .jpg, .png, .jpeg
    2. Video: .mp4, .mpeg
    3. Audio: .wav, .mp3, .m4a
    4. Text: .txt, .csv

    Args:
        uploaded_files (Union[str,Path,UploadedFile], optional): [description]. Defaults to session_state.upload_widget.
    """
    uploaded_files: Union[str, Path, UploadedFile] = uploaded_files
    if uploaded_files:
        start_time = perf_counter()
        if len(uploaded_files) == 1:
            filetype = get_filetype(uploaded_files[0])
            logger.info("Enter single")
            logger.info(filetype)

        else:
            logger.info("Enter multi")
            filetypes = map(compare_filetypes, zip(
                uploaded_files[:], uploaded_files[1:]))
            for check_result, filetype, file1, file2 in filetypes:
                if check_result:
                    logger.info("Filetype passed")
                    pass
                else:
                    filetype_error_msg = f"Filetype different for {file1} and {file2}"
                    logger.error(filetype_error_msg)
                    if field_placeholder:
                        field_placeholder["upload"].error(
                            filetype_error_msg)
                    break

                    # GET Filetype
        dataset.filetype = filetype.capitalize()

        end_time = perf_counter()
        time_elapsed = end_time - start_time
        number_of_files = len(uploaded_files)
        average_time = time_elapsed / number_of_files
        logger.info(
            f"Time taken to compare filetypes {time_elapsed}s with average of {average_time}s for {number_of_files}")


def check_args_kwargs(wildcards: Union[List, Dict] = None, func: Callable[..., Any] = None):

    if wildcards and func:
        assert len(wildcards) == len(signature(
            func).parameters), "Length of wildcards does not meet length of arguments required by callback function"


class NetChange(IntEnum):
    NoChange = 0
    Addition = 1
    Removal = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return NetChange[s]
        except KeyError:
            raise ValueError()


def find_net_change(initial_list: List, submitted_list: List) -> Tuple:
    """ Determine whether Element was removed or added into/from the list


    Args:
        initial_list (List): List of elements before modification
        submitted_list (List): List of elements after modification

    Returns:
        Tuple: ([List of added/removed elements] , NetChange IntEnum constant)
    """

    # Let Set 1 = Initial List
    # Let set 2 =Newly submitted List

    # set 1 - set 2 REMOVAL
    # {1,2,3,4,5,6,7,8} - {1,2,3,4,5} = {6, 7, 8}
    diff_12 = set(initial_list).difference(submitted_list)

    # set 2 - set 1 ADDITION
    # {1,2,3,4,5,6,7,8} - {1,2,3,4,5} = {6, 7, 8}
    diff_21 = set(submitted_list).difference(initial_list)

    # ****************** REMOVAL ***********************
    if diff_12:

        removed_elements = list(diff_12)
        flag = NetChange.Removal
        logger.info(f"REMOVAL: {flag}")

        return removed_elements, flag

    # ****************** ADDITION ***********************
    elif diff_21:
        added_elements = list(diff_21)
        flag = NetChange.Addition
        logger.info(f"ADDITION: {flag}")

        return added_elements, flag

    else:
        flag = NetChange.NoChange
        logger.info(f"No Change: {flag}")
        return None, flag


def list_available_cameras(num_check_ports: int):
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    # https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python

    working_ports = []
    available_ports = []
    # check one extra port because port starts from 0
    for dev_port in range(num_check_ports + 1):
        cap = cv2.VideoCapture(dev_port)
        if not cap.isOpened():
            logger.info(f"Port {dev_port} is not working.")
        is_reading, img = cap.read()
        w = cap.get(3)
        h = cap.get(4)
        if is_reading:
            logger.info(
                f"Port {dev_port} is working and reads images ({h} x {w})")
            working_ports.append(dev_port)
            # cv2.imshow('frame', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            logger.info(f"Port {dev_port} for camera ({h} x {w}) is present "
                        "but does not reads.")
            available_ports.append(dev_port)
        cap.release()
    return available_ports, working_ports


def save_image(frame: np.ndarray, save_dir: Path, channels: str = 'BGR',
               timezone: str = 'Singapore', prefix: str = None):
    """Optionally pass in `prefix` to prepend to the filename."""
    now = get_now_string(timezone=timezone)
    if prefix:
        filename = f'{prefix}-image-{now}.png'
    else:
        filename = f'image-{now}.png'
    save_path = str(save_dir / filename)
    logger.info(f'saving frame at: "{save_path}"')

    if channels == 'RGB':
        # OpenCV needs BGR format
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        out = frame
    cv2.imwrite(save_path, out)


def reset_camera():
    for k, cap in filter(lambda x: x[0].startswith('camera'), session_state.items()):
        if isinstance(cap, WebcamVideoStream):
            cap.stop()
        elif isinstance(cap, cv2.VideoCapture):
            # cv2.VideoCapture instance
            cap.release()
        del session_state[k]


def reset_camera_and_ports():
    reset_camera()
    if 'working_ports' in session_state:
        del session_state['working_ports']