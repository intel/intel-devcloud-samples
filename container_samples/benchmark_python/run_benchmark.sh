#!/usr/bin/env sh
#source /opt/intel/openvino_2021.2.185/bin/setupvars.sh 
#alias python3=python3.6
#$#alias python3=python3.6
OUTPUT_FILE="output_benchmark"
DEVICE="CPU"
FP_MODEL_32=models/FP32
FP_MODEL_16=models/FP16
API="async"
# Benchmark Application script writes output to a file inside a directory. We make sure that this directory exists.
#  The output directory is the first argument of the bash script

mkdir -p $OUTPUT_FILE



python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m ${FP_MODEL_32}/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder ${OUTPUT_FILE}
