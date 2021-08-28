FROM quay.io/devcloud/devcloud-openvino-data-dev:2021.4_latest

RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh 

# Install git 
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# install vlc player to play mp4 videos
RUN apt-get update \
&& apt-get install -y vlc



RUN pip install tensorflow==1.15.5

ENV USERNAME=intel
ENV PASSWORD=intel

RUN usermod -a -G  intel  intel

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD  developer-samples/python/tiny-yolo-v3 ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3
RUN chown -R  intel:intel  ${INTEL_OPENVINO_DIR} ${INTEL_OPENVINO_DIR}/python  ${INTEL_OPENVINO_DIR}/python/samples  ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3 ${INTEL_OPENVINO_DIR}/deployment_tools ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites  /var/lib/dpkg

RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py
RUN chmod -R 777 ${INTEL_OPENVINO_DIR}/python/samples/
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3/*.sh

USER intel

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE 

ARG PRECISION="FP16,FP32"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.582"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_tiny-yolo-v3_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="resnet-50-tf"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/tiny-yolo-v3/classroom.mp4"
ENV INPUT_FILE=$INPUT_FILE

RUN source  ${INTEL_OPENVINO_DIR}/bin/setupvars.sh 
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing run tiny-yolo-v3-python app using OpenVINO ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/tiny-yolo-v3/run_tiny_yolo_v3.sh"


