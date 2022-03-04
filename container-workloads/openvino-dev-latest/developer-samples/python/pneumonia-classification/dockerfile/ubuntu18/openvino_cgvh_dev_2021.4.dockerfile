#Copyright (C) 2022 Intel Corporation
#SPDX-License-Identifier: MIT

#Building openvino base image from public source
FROM openvino/ubuntu18_data_dev:2021.4.2
RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh

RUN apt-get update && \
    apt-get autoremove -y dpkg-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD developer-samples/python/pneumonia-classification  ${INTEL_OPENVINO_DIR}/python/samples/pneumonia-classification


RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py
RUN chmod -R 777 ${INTEL_OPENVINO_DIR}/python/samples/
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/pneumonia-classification/*.sh


ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.752"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG OUTPUT_FOLDER="output_pneumnia_classification"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER


ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

RUN source  ${INTEL_OPENVINO_DIR}/bin/setupvars.sh
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing object detection app using OpenVINO ......."
WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/pneumonia-classification
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/pneumonia-classification/run_pneumonia.sh"

