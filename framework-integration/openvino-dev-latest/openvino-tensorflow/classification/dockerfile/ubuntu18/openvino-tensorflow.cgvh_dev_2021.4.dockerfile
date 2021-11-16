FROM docker.io/openvino/ubuntu18_runtime:2021.4.1
 
USER root

ADD framework-integration/openvino-dev-latest/openvino-tensorflow/classification /classification-ovtf
ADD framework-integration/openvino-dev-latest/openvino-tensorflow/data /classification-ovtf/data
RUN chmod 0777 /classification-ovtf
RUN chgrp -R 0 /classification-ovtf && \
    chmod -R g=u /classification-ovtf

RUN apt update && apt -y install wget gcc-8 unzip libssl1.0.0 software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && apt update && apt -y install --only-upgrade libstdc++6

RUN ls
RUN pwd
RUN chmod 777 /classification-ovtf/*.sh

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

ARG INPUT_LAYER="input"
ENV INPUT_LAYER=$INPUT_LAYER

ARG OUTPUT_LAYER="InceptionV3/Predictions/Reshape_1"
ENV OUTPUT_LAYER=$OUTPUT_LAYER

ARG LABELS="imagenet_slim_labels.txt"
ENV LABELS=$LABELS

ARG MODEL="inception_v3_2016_08_28_frozen.pb"
ENV MODEL=$MODEL

ARG INPUT_HEIGHT=299
ENV INPUT_HEIGHT=$INPUT_HEIGHT

ARG INPUT_WIDTH=299
ENV INPUT_WIDTH=$INPUT_WIDTH

ARG OUTPUT_FOLDER="/mount_folder"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM


RUN echo "Executing classification sample using Intel Openvino Integration with Tensorflow  ......."

WORKDIR /classification-ovtf

RUN curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C data -xz
RUN chmod 777 /classification-ovtf/data/*
RUN chmod -R 777 /classification-ovtf

RUN pip3 install  pillow https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.1/tensorflow_abi1-2.5.1-cp36-cp36m-manylinux2010_x86_64.whl
RUN pip3 install ./data/openvino_tensorflow-1.0.1-cp36-cp36m-manylinux2014_x86_64.whl

ENTRYPOINT /bin/bash -c "source run_ovtf_classification.sh"


