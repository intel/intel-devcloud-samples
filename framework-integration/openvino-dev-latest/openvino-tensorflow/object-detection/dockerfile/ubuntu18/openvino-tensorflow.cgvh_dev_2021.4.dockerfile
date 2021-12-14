FROM docker.io/openvino/ubuntu18_runtime:2021.4.1

USER root

ADD framework-integration/openvino-dev-latest/openvino-tensorflow/object-detection /object-detection-ovtf

ADD framework-integration/openvino-dev-latest/openvino-tensorflow/data /object-detection-ovtf/data

RUN chmod 0777 /object-detection-ovtf
RUN chgrp -R 0 /object-detection-ovtf && \
    chmod -R g=u /object-detection-ovtf


RUN apt update && apt -y install python-opencv wget gcc-8 unzip libssl1.0.0 software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && apt update && apt -y install --only-upgrade libstdc++6


RUN ls
RUN pwd
RUN chmod 777 /object-detection-ovtf/*.sh

RUN echo "Intel devcloud Sample containerization begin ......."


ARG OPENVINO_VERSION="2021.4.689"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG FLAG="openvino"
ENV FLAG=$FLAG

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG INPUT_FILE="grace_hopper.jpg"
ENV INPUT_FILE="$INPUT_FILE"

ARG INPUT_TYPE="image"
ENV INPUT_TYPE=$INPUT_TYPE

ARG INPUT_LAYER="inputs"
ENV INPUT_LAYER=$INPUT_LAYER

ARG OUTPUT_LAYER="output_boxes"
ENV OUTPUT_LAYER=$OUTPUT_LAYER

ARG LABELS="coco.names"
ENV LABELS=$LABELS

ARG MODEL="yolo_v3_160.pb"
ENV MODEL=$MODEL

ARG INPUT_HEIGHT=160
ENV INPUT_HEIGHT=$INPUT_HEIGHT

ARG INPUT_WIDTH=160
ENV INPUT_WIDTH=$INPUT_WIDTH

ARG INPUT_WIDTH=160
ENV INPUT_WIDTH=$INPUT_WIDTH

ARG OUTPUT_FILENAME="detections.jpg"
ENV OUTPUT_FILENAME=$OUTPUT_FILENAME

ARG OUTPUT_DIRECTORY="/mount_folder"
ENV OUTPUT_DIRECTORY=$OUTPUT_DIRECTORY

ARG OUTPUT_FOLDER="/mount_folder"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM


RUN echo "Executing Object Detection sample using Intel Openvino Integration with Tensorflow  ......."
RUN apt-get update && apt-get install -y git wget

WORKDIR /object-detection-ovtf


RUN /bin/bash -c "source convert_yolov3_160.sh"

RUN pip3 install pillow https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.1/tensorflow_abi1-2.5.1-cp36-cp36m-manylinux2010_x86_64.whl
RUN pip3 install ./data/openvino_tensorflow-1.0.1-cp36-cp36m-manylinux2014_x86_64.whl


ENTRYPOINT /bin/bash -c "source run_ovtf_objectdetection.sh"


