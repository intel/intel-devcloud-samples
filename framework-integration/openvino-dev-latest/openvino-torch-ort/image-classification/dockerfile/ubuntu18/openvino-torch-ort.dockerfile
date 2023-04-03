ARG OPENVINO_VERSION=2022.2.0
FROM openvino/ubuntu18_runtime:${OPENVINO_VERSION}

ENV DEBIAN_FRONTEND noninteractive

USER root

COPY framework-integration/openvino-dev-latest/openvino-torch-ort/image-classification /classification-torch-ort
RUN chmod 0777 /classification-torch-ort
RUN chgrp -R 0 /classification-torch-ort && \
    chmod -R g=u /classification-torch-ort

WORKDIR /classification-torch-ort
RUN chmod 777 /classification-torch-ort/*.sh

RUN apt update; \ 
    apt-get install -y --no-install-recommends \
    python3.8 \
    libpython3.8-dev \
    curl \
    wget; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2;

RUN python3 -m pip install --upgrade pip && pip install -U numpy && pip install wheel && pip install -U setuptools

RUN pip install torch-ort-infer[openvino]

RUN pip install wget pandas transformers pillow torchvision

RUN python3 -m torch_ort.configure;

RUN wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG INPUT_FILE="data/plane.jpg"
ENV INPUT_FILE="$INPUT_FILE"

ARG LABELS="imagenet_classes.txt"
ENV LABELS=$LABELS

ARG PROVIDER="openvino"
ENV PROVIDER=$PROVIDER

ARG BACKEND="CPU"
ENV BACKEND=$BACKEND

ARG PRECISION="FP32"
ENV PRECISION=$PRECISION

ARG OUTPUT_FOLDER="/mount_folder"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

RUN echo "Executing image classification sample using Intel Openvino Integration with Torch-ORT  ......."

ENTRYPOINT /bin/bash -c "source run_ov_torch_ort_image_classification.sh"
