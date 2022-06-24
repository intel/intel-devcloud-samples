#!/usr/bin/env python3.7 
echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "****OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "Running on" $RUN_ON_PREM

FP16='FP16'
FP32='FP32'

sample_name="pneumonia-classification"
DEVICE=$DEVICE
NUM_REQS=2
Output_folder_32="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP32"
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"
XML_IR_FP32="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP32"
INPUT_FILE="./validation_images/NORMAL/*.jpeg"
API="async"

source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh
#pip3 install Pillow

if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder \$FP16"
   echo  $Output_folder_16
   mkdir -p $Output_folder_16
   echo $XML_IR_FP16
   mkdir -p $XML_IR_FP16

   # Create FP16 IR files
   mo  \
   --input_model model.pb \
   --input_shape=[1,224,224,3] \
   --data_type $FP16 \
   -o  $XML_IR_FP16\
   --mean_values [123.75,116.28,103.58] \
   --scale_values [58.395,57.12,57.375]

   #Run the pneumonia detection code for IR 16
   python3 classification_pneumonia.py -m $XML_IR_FP16/model.xml \
                                       -i "$INPUT_FILE" \
                                       -o $Output_folder_16  \
                                       -d $DEVICE
fi 




