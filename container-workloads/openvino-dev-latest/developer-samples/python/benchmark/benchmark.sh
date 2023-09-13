echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "MODEL:  $MODEL"
echo "DOWNLOADED MODEL PATH:  $DOWNLOADED_MODEL_PATH"
echo "Running on" $RUN_ON_PREM

FP16='FP16'
FP32='FP32'
DEVICE=$DEVICE
NUM_REQS=2
API="async"
Output_folder_32="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP32"
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"
XML_IR_FP32="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP32"


sample_name="benchmark"

MODEL="face-detection-retail-0005"
source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh 
mkdir -p $RUN_ON_PREM/models
#chmod -R 777 $RUN_ON_PREM
omz_downloader  --name $MODEL  -o $RUN_ON_PREM/models


if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder \$FP16"
   mkdir -p $Output_folder_16
   mkdir -p $XML_IR_FP16
   
   cp $RUN_ON_PREM/models/intel/$MODEL/FP16/* $XML_IR_FP16
   

   python3 /opt/intel/openvino_$OPENVINO_VERSION/python/samples/benchmark/benchmark_app.py -m $XML_IR_FP16/$MODEL.xml \
            -d $DEVICE \
            -niter 10 \
            -api $API \
            --report_type detailed_counters \
            --report_folder  $Output_folder_16

   cat $RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16/performance.txt

fi
