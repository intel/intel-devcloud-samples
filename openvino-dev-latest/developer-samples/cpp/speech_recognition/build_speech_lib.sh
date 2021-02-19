#!/usr/bin/env bash

# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ $OS_PATH == "x86_64" ]; then
  OS_PATH="intel64"
  ARCH="x64"
  NUM_THREADS="-j8"
fi

${INTEL_OPENVINO_DIR}/bin/setupvars.sh
src_dir="${INTEL_OPENVINO_DIR}/data_processing/audio/speech_recognition/"
build_dir="$PWD"
if [ -e $build_dir/CMakeCache.txt ]; then
    rm -rf $build_dir/CMakeCache.txt
fi

cmake $src_dir
cd "$build_dir"
make $NUM_THREADS

