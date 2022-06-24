#Copyright (C) 2022 Intel Corporation
#SPDX-License-Identifier: MIT

#Building openvino base image from public source

FROM openvino/ubuntu18_dev:2022.1.0
RUN pip install openvino-dev[caffe]

RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
#RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh 

# Install git 
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install tensorflow==1.15.5
#RUN pip install pillow==9.0.0
RUN pip install numpy==1.19.0


RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD  developer-samples/python/tiny-yolo-v3 ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3
RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples
ADD  developer-samples/python/tiny-yolo-v3 ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3
#RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py
RUN chmod -R 777 ${INTEL_OPENVINO_DIR}/python/samples/
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3/*.sh


ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE 

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2022.1.0.643"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_tiny-yolo-v3_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="resnet-50-tf"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/tiny-yolo-v3/classroom.mp4"
ENV INPUT_FILE=$INPUT_FILE


RUN source  /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh

RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing run tiny-yolo-v3-python app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3/run_tiny_yolo_v3.sh"


