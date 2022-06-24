#Copyright (C) 2022 Intel Corporation
#SPDX-License-Identifier: MIT

#Building openvino base image from public source  

FROM openvino/ubuntu18_dev:2022.1.0
RUN pip install openvino-dev[caffe]

RUN echo "OpenVINO installation done  ......."
RUN echo ${INTEL_OPENVINO_DIR}
USER root
RUN echo "Intel devcloud benchmak sample containerization begin ......."


RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples


ADD developer-samples/python/benchmark   ${INTEL_OPENVINO_DIR}/python/samples//benchmark

RUN chmod  -R 777 ${INTEL_OPENVINO_DIR}/python/samples


ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2022.1.0.643"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_benchmark_app_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

RUN source  /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh

RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing Benchmarking app using OpenVINO latest ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/benchmark
ENTRYPOINT  /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/benchmark/benchmark.sh"
                                                                                                        
                                                                         
