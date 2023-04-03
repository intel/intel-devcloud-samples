#Copyright (C) 2022 Intel Corporation
#SPDX-License-Identifier: MIT

#Building openvino base image from public source

FROM openvino/ubuntu20_dev:2022.3.0

RUN pip install openvino-dev[caffe]


RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root

RUN apt-get update && apt-get install ffmpeg -y
RUN chmod 0777 ${INTEL_OPENVINO_DIR}/python

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD developer-samples/python/safety-gear-detection ${INTEL_OPENVINO_DIR}/python/samples/safety-gear-detection

RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/safety-gear-detection/*.sh

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2022.3.0.9038"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_safety_gear_det_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="mobilenet-ssd"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/safety-gear-detection/resources/Safety_Full_Hat_and_Vest.mp4"
ENV INPUT_FILE=$INPUT_FILE

RUN source  /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh

RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing safety gear detection app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/safety-gear-detection
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/safety-gear-detection/run_safety_gear_detection.sh"
