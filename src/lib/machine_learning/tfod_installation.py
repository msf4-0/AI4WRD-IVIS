""" 
Title: TensorFlow Object Detection (TFOD) API Installation
Date: 30/9/2021 
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
import os
import shlex
import shutil
import stat
import sys
import wget
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

# >>>> User-defined Modules >>>>
from core.utils.log import logger
from path_desc import TFOD_DIR, chdir_root

PROTOC_PATH = TFOD_DIR.parent / 'protoc_installation'


def run_command(command_line_args):
    logger.info(f"Running command: '{command_line_args}'")
    if isinstance(command_line_args, str):
        # must pass in list to the subprocess when shell=False, which
        # is required to work properly in Linux
        command_line_args = shlex.split(command_line_args)
    process = subprocess.Popen(command_line_args, shell=False,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)
    for line in process.stdout:
        line = line.rstrip()
        # print the live stdout output from the script
        if line:
            print(line)
    process.wait()
    process.kill()
    return process.stdout


def install_protoc():
    logger.info(f"{PROTOC_PATH = }")
    if PROTOC_PATH.exists():
        shutil.rmtree(PROTOC_PATH)
    os.makedirs(PROTOC_PATH)

    logger.info("Downloading protobuf dependencies ...")
    protoc_version = "3.19.1"  # updated from 3.15.6 -> 3.19.1
    protoc_zipfilename = f"protoc-{protoc_version}-win64.zip"
    url = f"https://github.com/protocolbuffers/protobuf/releases/download/v{protoc_version}/{protoc_zipfilename}"
    # download the protoc zipfile into the desired path, PROTOC_PATH
    wget.download(url, str(PROTOC_PATH))
    # unzip the zip file
    os.chdir(PROTOC_PATH)
    if os.name == "posix":
        run_command(f'unzip {protoc_zipfilename}')
    else:
        run_command(f'tar -xf {protoc_zipfilename}')
    chdir_root()
    os.remove(PROTOC_PATH / protoc_zipfilename)

    # add the path of $PROTOC_PATH/bin into the PATH in environment variable
    # to be able to run `protoc` as a command in terminal
    os.environ['PATH'] += os.pathsep + str((PROTOC_PATH / 'bin').resolve())


def del_rw(action, name, exc):
    """To delete .git directory in TFOD"""
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)


def install():
    for path in (TFOD_DIR, PROTOC_PATH):
        os.makedirs(path, exist_ok=True)

    # remove any existing TFOD stuff
    if TFOD_DIR.exists():
        shutil.rmtree(TFOD_DIR, onerror=del_rw)
    os.makedirs(TFOD_DIR)

    logger.info(
        "Cloning TFOD API from https://github.com/tensorflow/models...")
    run_command(
        f'git clone https://github.com/tensorflow/models "{TFOD_DIR}"')
    # reset to a commit for TensorFlow 2.7.0 right before update to 2.8.0
    # https://github.com/tensorflow/models/commit/cd21e8ff34b4e389fcdddf04045b14aad8c8a91b
    os.chdir(TFOD_DIR)
    run_command(
        'git reset --hard cd21e8ff34b4e389fcdddf04045b14aad8c8a91b')
    chdir_root()

    # Install Tensorflow Object Detection and dependencies such as protobuf and protoc
    # NOTE: Install COCO API ONLY if you want to perform evaluation
    if os.name == 'posix':
        logger.info('Installing COCO API ...')
        cwd = os.getcwd()
        run_command("git clone https://github.com/cocodataset/cocoapi.git")
        os.chdir("cocoapi/PythonAPI")
        run_command("make")
        run_command(f'cp -r pycocotools "{TFOD_DIR / "research"}"')
        os.chdir(cwd)
        # removed the unused cloned files from COCO API
        shutil.rmtree("cocoapi")
    elif os.name == 'nt':
        # 'nt' is for Windows
        # NOTE: Windows need to install COCO API first, but there is currently an ongoing issue
        # for running COCO evaluation on Windows, refer to here https://github.com/google/automl/issues/487
        # Instructions for COCO API Installation: https://github.com/ansonnn07/object-detection#coco-api-installation
        #  (under the COCO API installation section)
        # NOTE: using this new repo created to fix Windows installation for now
        # run_command(
        #     "pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI")
        pass

    install_protoc()

    # run the `protoc` command and install all the dependencies for TFOD API
    logger.info("Installing TFOD API ...")
    os.chdir(TFOD_DIR / "research")
    run_command('protoc object_detection/protos/*.proto --python_out=.')
    shutil.copy2('object_detection/packages/tf2/setup.py', './setup.py')
    # REMOVED --use-feature=2020-resolver as the new pip use this by default now
    # https://pip.pypa.io/en/stable/news/#deprecations-and-removals
    # https://github.com/pypa/pip/pull/11493
    run_command('pip install .')
    chdir_root()

    # run the requirements installation again to make sure all the versions are correct
    run_command('pip install -r requirements_no_hash.txt')

    # remove unnecessary files
    folders_to_del = (TFOD_DIR, (TFOD_DIR / 'research'))
    folders_to_keep = set(('research', 'object_detection', 'pycocotools'))
    for folder in folders_to_del:
        for path in folder.iterdir():
            if path.is_dir():
                # these folders should not be removed
                if path.name in folders_to_keep:
                    continue
                if path.name == '.git':
                    # .git folder needs special permission
                    shutil.rmtree(path, onerror=del_rw)
                else:
                    shutil.rmtree(path)
            else:
                os.remove(path)

    # NOTE: not running verification script for now as the environment might not be updated
    #   with the latest packages yet
    # VERIFICATION_SCRIPT = TFOD_DIR / 'research' / \
    #     'object_detection' / 'builders' / 'model_builder_tf2_test.py'
    # # Verify all the installation above works for TFOD API
    # logger.info("Verifying all works ...")
    # run_command(f"python {VERIFICATION_SCRIPT}")
    # logger.info(
    #     "If the last line above returns 'OK' then TFOD API is installed successfully.")

    chdir_root()


if __name__ == '__main__':
    install()
