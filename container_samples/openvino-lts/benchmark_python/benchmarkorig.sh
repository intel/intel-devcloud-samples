#!/usr/bin/bash 
#openvino_ver=openvino_2021.3.394
openvino_ver=openvino_2020.3.355
echo "openvino version"
echo $openvino_ver
source /opt/intel/$openvino_ver/bin/setupvars.sh
python3 -V

mkdir -p data/output_benchmark/IR/FP32
mkdir -p data/output_benchmark/IR/FP16
mkdir -p data/output_benchmark/results/IR32
mkdir -p data/output_benchmark/results/IR16
mkdir -p data/models

source /opt/intel/$openvino_ver/bin/setupvars.sh 
#alias python3=python3.6
python3 -V
python3 /opt/intel/$openvino_ver/deployment_tools/tools/model_downloader/downloader.py --name resnet-50-tf -o data/models


python3 /opt/intel/$openvino_ver/deployment_tools/model_optimizer/mo.py \
 --input_model data/models/public/resnet-50-tf/resnet_v1-50.pb \
 --input_shape=[1,224,224,3] \
 --mean_values=[123.68,116.78,103.94] \
 -o  data/output_benchmark/IR/FP16 \
 --data_type FP16

python3 /opt/intel/$openvino_ver/deployment_tools/model_optimizer/mo.py \
 --input_model data/models/public/resnet-50-tf/resnet_v1-50.pb \
 --input_shape=[1,224,224,3] \
 --mean_values=[123.68,116.78,103.94] \
 -o data/output_benchmark/IR/FP32 \
 --data_type FP32


OUTPUT_FILE="output_benchmark"
DEVICE="CPU"
FP_MODEL_32=data/output_benchmark/IR/FP32
FP_MODEL_16=data/output_benchmark/IR/FP16 
API="async"
# Benchmark Application script writes output to a file inside a directory. We make sure that this directory exists.
#  The output directory is the first argument of the bash script

OUTPUT_FILE_32=data/output_benchmark/results/IR32
OUTPUT_FILE_16=data/output_benchmark/results/IR16



#For IR 32
python3 ./benchmarkApp_python/benchmark_app.py -m ${FP_MODEL_32}/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder ${OUTPUT_FILE_32}

#For IR 16
python3 ./benchmarkApp_python/benchmark_app.py -m ${FP_MODEL_16}/resnet_v1-50.xml \
            -d $DEVICE
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder ${OUTPUT_FILE_16}
