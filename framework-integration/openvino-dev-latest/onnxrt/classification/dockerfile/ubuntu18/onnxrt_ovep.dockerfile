ARG OPENVINO_VERSION=2022.1.0
# Build stage
FROM openvino/ubuntu18_runtime:${OPENVINO_VERSION} AS builder

ENV WORKDIR_PATH=/home/openvino
WORKDIR $WORKDIR_PATH
ENV DEBIAN_FRONTEND noninteractive
ENV InferenceEngine_DIR=${INTEL_OPENVINO_DIR}/runtime/cmake

USER root

ADD framework-integration/openvino-dev-latest/onnxrt/classification /classification-onnxrt
ADD framework-integration/openvino-dev-latest/onnxrt/classification/data /classification-onnxrt/data
ADD framework-integration/openvino-dev-latest/onnxrt/classification/include /classification-onnxrt/include
ADD framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries /classification-onnxrt/ort-Libraries
ADD framework-integration/openvino-dev-latest/onnxrt/classification/CMakeLists.txt /classification-onnxrt/CMakeLists.txt
ADD framework-integration/openvino-dev-latest/onnxrt/classification/squeezenet_cpp_app.cpp /classification-onnxrt/squeezenet_cpp_app.cpp


RUN chmod 0777 /classification-onnxrt
RUN chgrp -R 0 /classification-onnxrt && \
    chmod -R g=u /classification-onnxrt

#Setup opencv
RUN cd /opt/intel/openvino_2022/extras/scripts && \
    ./download_opencv.sh

WORKDIR /classification-onnxrt

RUN chmod 777 /classification-onnxrt/*.sh
RUN echo "Intel devcloud Sample containerization begin ......."

ARG DEVICE="CPU_FP32"
ENV DEVICE=$DEVICE

ARG INPUT_FILE="demo.jpeg"
ENV INPUT_FILE="$INPUT_FILE"

ARG MODEL="squeezenet1.1-7.onnx"
ENV MODEL=$MODEL

ARG LABELS="synset.txt"
ENV LABELS=$LABELS

ARG EXECUTION_PROVIDER="--use_openvino"
ENV EXECUTION_PROVIDER=$EXECUTION_PROVIDER



RUN echo "Executing classification sample using Intel Openvino Integration with ONNXRT  ......."
ENTRYPOINT /bin/bash -c "source run_onnxrt_classification.sh"


