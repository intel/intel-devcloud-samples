FROM devcloud_data_dev:2021.4_latest

RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
RUN chmod 0777 ${INTEL_OPENVINO_DIR}/python

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD developer-samples/python/object-detection ${INTEL_OPENVINO_DIR}/python/samples/object-detection

RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/object-detection/*.sh

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16,FP32"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.582"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_obj_det_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="mobilenet-ssd"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/object-detection/cars_1900.mp4"
ENV INPUT_FILE=$INPUT_FILE

RUN source  ${INTEL_OPENVINO_DIR}/bin/setupvars.sh
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing object detection app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/object-detection
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/object-detection/run_object_detection.sh"
