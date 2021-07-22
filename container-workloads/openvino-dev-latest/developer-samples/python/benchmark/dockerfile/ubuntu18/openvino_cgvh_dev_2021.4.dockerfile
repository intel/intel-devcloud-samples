FROM quay.io/devcloud/devcloud-openvino-data-dev:2021.4_latest

RUN echo "OpenVINO installation done  ......."
RUN echo ${INTEL_OPENVINO_DIR}

RUN echo "Intel devcloud benchmak sample containerization begin ......."

USER root
RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh

ENV USERNAME=intel
ENV PASSWORD=intel
RUN usermod -a -G  intel  intel

RUN mkdir -p  ${INTEL_OPENVINO_DIR}/python/samples

ADD developer-samples/python/benchmark ${INTEL_OPENVINO_DIR}/python/samples/benchmark
ADD developer-samples/python/benchmark/benchmark ${INTEL_OPENVINO_DIR}/python/python3.7/openvino/tools/benchmark
ADD developer-samples/python/benchmark/benchmark ${INTEL_OPENVINO_DIR}/python/python3.6/openvino/tools/benchmark
COPY developer-samples/python/benchmark/main.py ${INTEL_OPENVINO_DIR}/python/python3.7/openvino/tools/benchmark/main.py
COPY developer-samples/python/benchmark/main.py ${INTEL_OPENVINO_DIR}/python/python3.6/openvino/tools/benchmark/main.py
COPY developer-samples/python/benchmark/main.py ${INTEL_OPENVINO_DIR}/python/python3.6/openvino/tools/benchmark/main.py
COPY developer-samples/python/benchmark/benchmark.sh ${INTEL_OPENVINO_DIR}/python/samples/benchmark
RUN chown -R  intel:intel  ${INTEL_OPENVINO_DIR} ${INTEL_OPENVINO_DIR}/python  ${INTEL_OPENVINO_DIR}/python/samples  ${INTEL_OPENVINO_DIR}/deployment_tools ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites  /var/lib/dpkg ${INTEL_OPENVINO_DIR}/python/samples/benchmark

RUN chmod 777 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/benchmark
RUN chmod 777 ${INTEL_OPENVINO_DIR}/python/samples/benchmark/*.sh

USER intel

ENV PATH ${INTEL_OPENVINO_DIR}/python/samples:$PATH

ARG DEVICE="CPU"
ENV DEVICE=$DEVICE

ARG PRECISION="FP16,FP32"
ENV PRECISION="$PRECISION"

ARG OPENVINO_VERSION="2021.4.582"
ENV OPENVINO_VERSION=$OPENVINO_VERSION

ARG OUTPUT_FOLDER="output_benchmark_app_latest"
ENV OUTPUT_FOLDER=$OUTPUT_FOLDER

ARG SUPPORTED_HW="CPU_TDP_70_205W"
ENV SUPPORTED_HW=$SUPPORTED_HW

ARG MODEL="resnet-50-tf"
ENV MODEL=$MODEL

ARG RUN_ON_PREM="/mount_folder"
ENV RUN_ON_PREM=$RUN_ON_PREM

RUN source  ${INTEL_OPENVINO_DIR}/bin/setupvars.sh
RUN echo "Generating OpenVINO IR files ......."
RUN echo "Executing Benchmarking app using OpenVINO latest ......."

WORKDIR ${INTEL_OPENVINO_DIR}/python/samples/benchmark
ENTRYPOINT  /bin/bash -c "source ${INTEL_OPENVINO_DIR}/python/samples/benchmark/benchmark.sh"
                                                                                                        
                                                                         
