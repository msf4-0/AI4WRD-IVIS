"""
Title: Helper
Date: 19/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import mimetypes
import sys
from collections import namedtuple
from enum import IntEnum
from functools import wraps
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import pandas as pd
import streamlit as st
from color_extract import color_extract
from colorutils import hex_to_hsv
from streamlit.uploaded_file_manager import UploadedFile

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import db_fetchone, init_connection
# >>>> User-defined Modules >>>>
from path_desc import chdir_root

conn = init_connection(**st.secrets["postgres"])


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


def split_string(string: str) -> List:

    # Split the string based on space delimiter
    list_string = string.split(' ')

    return list_string


def join_string(list_string: List) -> str:

    # Join the string based on '-' delimiter
    string = '-'.join(list_string)

    return string


def get_directory_name(name: str) -> str:
    directory_name = join_string(split_string(str(name))).lower()
    return directory_name


def is_empty(iterable: Union[List, Dict, set]) -> bool:
    return not bool(iterable)


# @st.cache
def create_dataframe(data: Union[List, Dict, pd.Series], column_names: List = None, sort: bool = False, sort_by: Optional[str] = None, asc: bool = True, date_time_format: bool = False) -> pd.DataFrame:
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


def dataframe2dict(orient='index'):

    def inner(func) -> List[dict]:
        @wraps(func)
        def convert_to_dict(*args, **kwargs):

            if args:
                df = func(*args)

            elif kwargs:
                df = func(**kwargs)

            dataframe_dict = list(df.to_dict(orient=orient).values())

            return dataframe_dict
        return convert_to_dict
    return inner


def datetime_formatter(data_list: Union[List[namedtuple], List[dict]], return_dict: bool = False) -> List:
    """Convert datetime format to %Y-%m-%d %H:%M:%S for Dict and namedtuple from DB query

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


def get_identifier_str_IntEnum(identifier: Union[str, IntEnum],
                               enumerator_class: IntEnum, identifier_dictionary: Dict,
                               string: bool = False):

    if string:

        # Get String form if deployment_type is type IntEnum class
        if isinstance(identifier, enumerator_class):
            identifier = [
                k for k, v in identifier_dictionary.items() if v == identifier][0]

    else:
        # Get IntEnum class constant if deployment_type is string
        if isinstance(identifier, str):
            identifier = identifier_dictionary[identifier]

    log_info(f"Deployment Type is :{identifier}")

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
        log_info(f"File: {file}")
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
    log_info(f"File tuple:{file_tuple}")
    if filetype1 == filetype2:
        log_info(f"File types:{filetype1,filetype2}")
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
            log_info("Enter single")
            log_info(filetype)

        else:
            log_info("Enter multi")
            filetypes = map(compare_filetypes, zip(
                uploaded_files[:], uploaded_files[1:]))
            for check_result, filetype, file1, file2 in filetypes:
                if check_result:
                    log_info("Filetype passed")
                    pass
                else:
                    filetype_error_msg = f"Filetype different for {file1} and {file2}"
                    log_error(filetype_error_msg)
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
        log_info(
            f"Time taken to compare filetypes {time_elapsed}s with average of {average_time}s for {number_of_files}")


def check_args_kwargs(wildcards: Union[List, Dict] = None, func: Callable[..., Any] = None):

    if wildcards and func:
        assert len(wildcards) == len(signature(
            func).parameters), "Length of wildcards does not meet length of arguments required by callback function"
