#-------------------------------------------------------------------------
# Copyright(C) 2021 Intel Corporation.
# SPDX-License-Identifier: MIT
#--------------------------------------------------------------------------
# Build stage
ARG OPENVINO_VERSION="2022.1.0"
FROM openvino/onnxruntime_ep_ubuntu18:${OPENVINO_VERSION}
USER root

ENV WORKDIR_PATH=/home/openvino
WORKDIR $WORKDIR_PATH
ENV DEBIAN_FRONTEND noninteractive
ARG DEVICE=CPU_FP32
ARG OPENVINO_VERSION

ENV InferenceEngine_DIR=${INTEL_OPENVINO_DIR}/runtime/cmake

#Setup opencv
RUN apt update && apt -y install python-opencv && \
    cd /opt/intel/openvino_2022/extras/scripts && \
    ./download_opencv.sh
ADD framework-integration/openvino-dev-latest/onnxrt/object-detection  /object-detection-onnxrt
ADD framework-integration/openvino-dev-latest/onnxrt/object-detection/data /object-detection-onnxrt/data
RUN chmod 0777 /object-detection-onnxrt
RUN chgrp -R 0 /object-detection-onnxrt && \
    chmod -R g=u /object-detection-onnxrt

RUN chmod 777 /object-detection-onnxrt/*

RUN echo "Intel devcloud Sample containerization begin ......."

ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG DEVICE="CPU_FP32"
ENV DEVICE=$DEVICE

ARG INPUT_FILE="manufacture0.mp4"
ENV INPUT_FILE="$INPUT_FILE"

ARG MODEL="Tiny_YoloV2_Cleanroom.onnx"
ENV MODEL=$MODEL

RUN echo "Executing object detection sample using Intel Openvino Integration with ONNXRT  ......."

WORKDIR /object-detection-onnxrt

ENTRYPOINT /bin/bash -c "source run_onnx_objectdetection.sh"
