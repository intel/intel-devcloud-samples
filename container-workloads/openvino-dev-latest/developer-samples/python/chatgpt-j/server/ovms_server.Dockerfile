# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
FROM openvino/model_server
USER root
RUN mkdir -p /onnx/1
ADD ./onnx/1/* ./onnx/1/


