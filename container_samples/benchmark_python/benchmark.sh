#!/usr/bin/env python3.6 
source /opt/intel/openvino_2021.2.185/bin/setupvars.sh 
alias python3=python3.6
python3 -V
python3 /opt/intel/openvino_2021.2.185/deployment_tools/tools/model_downloader/downloader.py --name resnet-50-tf -o models

mkdir -p models/FP32
mkdir -p models/FP16

python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py \
 --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
 --input_shape=[1,224,224,3] \
 --mean_values=[123.68,116.78,103.94] \
 -o models/FP16 \
 --data_type FP16

python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py \
 --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
 --input_shape=[1,224,224,3] \
 --mean_values=[123.68,116.78,103.94] \
 -o models/FP32 \
 --data_type FP32


bash run_benchmark.sh 
