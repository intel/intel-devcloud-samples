#!/usr/bin/env python3.7 

echo "$(cat $1)"
device=$(cut -d "," -f 1  $1 |xargs)
FP16=$(cut -d "," -f 2  $1 | xargs)
FP32=$(cut -d "," -f 3  $1 | xargs)
ver=$(cut -d "," -f 4  $1 | xargs)
openvino_ver=openvino_$ver

echo "Openvino Package Version: $openvino_ver"
echo "Device:  $device"
echo "Precision:  $FP16, $FP32"


sample_name="benchmark"
DEVICE=$device
NUM_REQS=2
Output_folder_32="data/output_$sample_name/results/FP32"
Output_folder_16="data/output_$sample_name/results/FP16"
XML_IR_FP16="data/output_$sample_name/IR/FP16"
XML_IR_FP32="data/output_$sample_name/IR/FP32"

API="async"

source /opt/intel/$openvino_ver/bin/setupvars.sh 

python3 /opt/intel/$openvino_ver/deployment_tools/tools/model_downloader/downloader.py --name resnet-50-tf -o models

if [[ ! -z "$FP16" ]];
then
   echo "Creating output folder \$FP16"
   mkdir -p $Output_folder_16
   mkdir -p $XML_IR_FP16
   python3 /opt/intel/$openvino_ver/deployment_tools/model_optimizer/mo.py \
   --data_type $FP16 \
   --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
   --input_shape=[1,224,224,3] \
   --mean_values=[123.68,116.78,103.94] \
   --output_dir $XML_IR_FP16 \


   python3 ./benchmarkApp_python/benchmark_app.py -m $XML_IR_FP16/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder $Output_folder_16

fi

if [[ ! -z "$FP32" ]];
then
   echo "Creating output folder \$FP32"
   mkdir -p $Output_folder_32
   mkdir -p $XML_IR_FP32
   python3 /opt/intel/$openvino_ver/deployment_tools/model_optimizer/mo.py \
   --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
   --input_shape=[1,224,224,3] \
   --mean_values=[123.68,116.78,103.94] \
   --data_type $FP32 \
   --output_dir $XML_IR_FP32 \


   #For IR 32
   python3 ./benchmarkApp_python/benchmark_app.py -m $XML_IR_FP32/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder $Output_folder_32
fi

