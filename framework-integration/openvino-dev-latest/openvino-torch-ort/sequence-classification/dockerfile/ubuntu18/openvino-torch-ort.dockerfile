ARG OPENVINO_VERSION=2022.2.0
FROM openvino/ubuntu18_runtime:${OPENVINO_VERSION}

ENV DEBIAN_FRONTEND noninteractive

USER root

COPY framework-integration/openvino-dev-latest/openvino-torch-ort/sequence-classification /sequence-classification-torch-ort
RUN chmod 0777 /sequence-classification-torch-ort
RUN chgrp -R 0 /sequence-classification-torch-ort && \
    chmod -R g=u /sequence-classification-torch-ort

WORKDIR /sequence-classification-torch-ort
RUN chmod 777 /sequence-classification-torch-ort/*.sh

RUN apt update; \ 
    apt-get install -y --no-install-recommends \
    python3.8 \
    libpython3.8-dev \
    curl \
    wget; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2;

RUN apt install -y python3-pip && python3 -m pip install --upgrade pip

RUN pip install torch-ort-infer[openvino]

RUN pip install wget pandas transformers pillow torchvision

RUN python3 -m torch_ort.configure;

ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG INPUT="Replace me with any text you'd like ."
ENV INPUT="$INPUT"

ARG INPUT_FILE=""
ENV INPUT_FILE="$INPUT_FILE"

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

RUN echo "Executing sequence classification sample using Intel Openvino Integration with Torch-ORT  ......."

ENTRYPOINT /bin/bash -c "source run_ov_torch_ort_sequence_classification.sh"
