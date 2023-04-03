# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
FROM openvino/ubuntu20_data_runtime
USER root
COPY . /application
WORKDIR /application
ENV http_proxy="http://proxy-chain.intel.com:911"
ENV https_proxy="http://proxy-chain.intel.com:912"
RUN apt-get update && apt-get -y upgrade && apt-get autoremove -y
RUN pip3 install -r client_requirements.txt
ENV http_proxy=""
ENV https_proxy=""
ENTRYPOINT ["bash","./run.sh"]
