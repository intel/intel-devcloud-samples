echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "MODEL:  $MODEL"
echo "Running on" $RUN_ON_PREM

#RUN_ON_PREM="data"

FP16='FP16'
FP32='FP32'
#INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/object-detection-python/cars_1900.mp4"
DEVICE=$DEVICE
NUM_REQS=2
API="async"
Output_folder_32="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP32"
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"
XML_IR_FP32="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP32"

IR_FP16="$XML_IR_FP16/mobilenet-ssd.xml"
IR_FP32="$XML_IR_FP32/mobilenet-ssd.xml"


sample_name="benchmark"

source /opt/intel/openvino_$OPENVINO_VERSION/bin/setupvars.sh 
python3 -V

python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/tools/model_downloader/downloader.py --name resnet-50-tf -o models


if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder \$FP16"
   mkdir -p $Output_folder_16
   mkdir -p $XML_IR_FP16
   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
   --data_type $FP16 \
   --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
   --input_shape=[1,224,224,3] \
   --mean_values=[123.68,116.78,103.94] \
   --output_dir $XML_IR_FP16 \


   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/tools/benchmark_tool/benchmark_app.py -m $XML_IR_FP16/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder $Output_folder_16

fi

if [[ "$PRECISION" == *"$F32"* ]];
then
   echo "Creating output folder \$FP32"
   mkdir -p $Output_folder_32
   mkdir -p $XML_IR_FP32
   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
   --input_model models/public/resnet-50-tf/resnet_v1-50.pb \
   --input_shape=[1,224,224,3] \
   --mean_values=[123.68,116.78,103.94] \
   --data_type $FP32 \
   --output_dir $XML_IR_FP32 \


   #For IR 32
   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/tools/benchmark_tool/benchmark_app.py -m $XML_IR_FP32/resnet_v1-50.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder $Output_folder_32
fi
