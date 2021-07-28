FROM quay.io/devcloud/devcloud-openvino-data-dev:2021.4_latest

RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
RUN chmod 0777 ${INTEL_OPENVINO_DIR}/python

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/tutorials

ADD tutorials/python/dlstreamer ${INTEL_OPENVINO_DIR}/python/tutorials/dlstreamer

RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/tutorials/dlstreamer/*.sh

ENV PATH ${INTEL_OPENVINO_DIR}/python/tutorials:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.582"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_dlstreamer_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="${INTEL_OPENVINO_DIR}/python/tutorials/dlstreamer/cars_1900.mp4"
ENV INPUT_FILE=$INPUT_FILE

RUN source  ${INTEL_OPENVINO_DIR}/bin/setupvars.sh
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing object detection app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/tutorials/dlstreamer
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/tutorials/dlstreamer/run_dlstreamer_tutorial.sh"
