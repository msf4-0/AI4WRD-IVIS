# [Choice] Python version: 3, 3.8, 3.7, 3.6
ARG VARIANT=3.8
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

ENV PYTHONUNBUFFERED 1
# this DOCKERCONTAINER is to tell the database_setup script to setup for Docker container
ENV DOCKERCONTAINER 1

# Update args in docker-compose.yaml to set the UID/GID of the "vscode" user.
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then groupmod --gid $USER_GID vscode && usermod --uid $USER_UID --gid $USER_GID vscode; fi

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Clone our repository
# RUN git clone https://github.com/msf4-0/Integrated-Vision-Inspection-System.git
# # Change into our root project directory
# WORKDIR /Integrated-Vision-Inspection-System/

# # [Optional] If your requirements rarely change, uncomment this section to add them to the image.
# COPY requirements_no_hash.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#     && rm -rf /tmp/pip-tmp

# install TensorFlow Object Detection API (TFOD)
# RUN python src/lib/machine_learning/tfod_installation.py

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>


# NOTE: NVIDIA Container Toolkit will need to be installed on your system
# to have access to GPU in the Docker container. Refer
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

