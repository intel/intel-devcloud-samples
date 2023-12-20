#Copyright (C) 2023 Intel Corporation
#SPDX-License-Identifier: MIT

#Building openvino base image from public source
FROM openvino/ubuntu20_dev:2023.2.0
RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root

RUN chmod 777 ${INTEL_OPENVINO_DIR}/python

# Install git 
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install openvino-dev[caffe]
RUN pip install ultralytics==8.0.123
RUN pip install numpy
RUN pip install opencv-python
RUN pip install Pillow

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples
ADD  developer-samples/python/yolo-v8 ${INTEL_OPENVINO_DIR}/python/samples/yolo-v8

RUN chmod -R 777 ${INTEL_OPENVINO_DIR}/python/samples/
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/yolo-v8/*.sh

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE 

ARG PRECISION="FP32"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2023"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_yolo-v8_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="yolo-v8"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

RUN echo "Executing run yolo-v8-python app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/yolo-v8
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/yolo-v8/run_yolo_v8.sh"
