#!/usr/bin/env python3.7

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "MODEL:  $MODEL"
echo "Running on" $RUN_ON_PREM
echo "Input image" $INPUT_FILE


FP16='FP16'
FP32='FP32'
DEVICE=$DEVICE
NUM_REQS=2
Output_folder_32="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP32"
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"
XML_IR_FP32="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP32"

IR_FP16="$XML_IR_FP16/mobilenet-ssd.xml"
IR_FP32="$XML_IR_FP32/mobilenet-ssd.xml"

Sample_name="object_detection.py"

SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution



source /opt/intel/openvino_$OPENVINO_VERSION/bin/setupvars.sh

mkdir -p $RUN_ON_PREM/raw_models/public

#python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/tools/model_downloader/downloader.py  --name $MODEL -o $RUN_ON_PREM/raw_models 

cp -r mobilenet-ssd $RUN_ON_PREM/raw_models/public

if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder \$FP16"
   mkdir -p $Output_folder_16 
   mkdir -p $XML_IR_FP16 
   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
   --input_model $RUN_ON_PREM/raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
   --data_type $FP16 \
   --output_dir $XML_IR_FP16 \
   --scale 256 \
   --mean_values [127,127,127]

   python3 $Sample_name  -i $INPUT_FILE  -m  $IR_FP16  --labels labels.txt -o $Output_folder_16 -d $DEVICE -nireq $NUM_REQS

   python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $Output_folder_16 \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION


fi

if [[ "$PRECISION" == *"$FP32"* ]];
then
   echo "Creating output folder \$FP32"
   mkdir -p $Output_folder_32
   mkdir -p $XML_IR_FP32
   python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
   --input_model $RUN_ON_PREM/raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
   --data_type $FP32 \
   --output_dir $XML_IR_FP32 \
   --scale 256 \
   --mean_values [127,127,127]

   python3 $Sample_name  -i $INPUT_FILE  -m  $IR_FP32  --labels labels.txt -o $Output_folder_32 -d $DEVICE -nireq $NUM_REQS

   python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $Output_folder_32 \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION

fi



