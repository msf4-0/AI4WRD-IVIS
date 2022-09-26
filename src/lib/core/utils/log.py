"""
Title: Logger
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from natsort import os_sorted

# environment variable should always be string
if os.getenv('DEBUG', '1') == '1':
    # added module name, function name, and also line number
    FORMAT = '[%(levelname)s] %(asctime)s - [%(module)s.%(funcName)s: %(lineno)d] %(message)s'
    LEVEL = logging.DEBUG
else:
    FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
    LEVEL = logging.INFO

LOG_FILE_DATEFMT = '%Y-%m-%d_%H-%M-%S_%f'
LOG_START_TIME = datetime.now().strftime(LOG_FILE_DATEFMT)
# Linux: /home/<username>/vision_system_logs
# Windows: C:/Users/<username>/vision_system_logs
LOG_DIR = Path.home() / 'vision_system_logs'
LOG_FILE = LOG_DIR / f'logs_{LOG_START_TIME}.txt'
if not LOG_DIR.exists():
    os.makedirs(LOG_DIR)

# change this accordingly
MAX_LOGFILES_TO_KEEP = 10
existing_logfiles = list(LOG_DIR.iterdir())
total_logfiles = len(existing_logfiles)
if total_logfiles > MAX_LOGFILES_TO_KEEP:
    sorted_logfiles = os_sorted(existing_logfiles)

    total_to_del = total_logfiles - MAX_LOGFILES_TO_KEEP
    for i, fpath in enumerate(sorted_logfiles, start=1):
        # print(f"Deleting old logfile: {fpath}")
        try:
            fpath.unlink()
        except Exception as e:
            print(f"Failed to delete the old logfile: {e}")
        if i == total_to_del:
            break

# create logger
logger = logging.getLogger(__name__)
# set log level for all handlers
logger.setLevel(LEVEL)

# create formatter
DATEFMT = '%d-%b-%y %H:%M:%S'
formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

# create console handler and setup level & formatter
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(LEVEL)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

# create file handler and setup for logger
print(f"[INFO] Logging console outputs to {LOG_FILE}")
fileHandler = RotatingFileHandler(
    str(LOG_FILE), maxBytes=300_000, backupCount=5)
fileHandler.setLevel(LEVEL)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# must set this to False to make sure that child logger does not propagate
#  its message to the root logger
logger.propagate = False

# It's not recommend to use the functions below as the logger will not be able to tell
# where the logger command was called, e.g. the `filename` will always be this script: log.py
# rather than the filename where the logger was called outside this script.


def std_log(msg):
    logger.info(msg)


def log_debug(msg):
    logger.debug(msg)


def log_info(msg):
    logger.info(msg)


def log_error(msg):
    logger.error(msg)


def log_warning(msg):
    logger.warning(msg)
