#Building openvino base image from public source
FROM openvino/ubuntu18_data_dev:2021.4.2

RUN echo "OpenVINO installation done  ......."
RUN echo "Intel devcloud Sample containerization begin ......."

USER root
RUN chmod 0777 ${INTEL_OPENVINO_DIR}/python

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD developer-samples/python/people-counter ${INTEL_OPENVINO_DIR}/python/samples/people-counter

RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/people-counter/*.sh

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.752"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_people_counter_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="person-detection-retail-0013"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

ARG INPUT_FILE="resources/Pedestrain_Detect_2_1_1.mp4"
ENV INPUT_FILE=$INPUT_FILE

RUN ["/bin/bash", "-c", "source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh"]
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing object detection app using OpenVINO ......."
WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/people-counter
ENTRYPOINT /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/people-counter/run_people_counter.sh"
