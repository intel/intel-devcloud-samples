#!/usr/bin/env python3.7

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "MODEL:  $MODEL"
echo "Running on" $RUN_ON_PREM
echo "Input image" $INPUT_FILE


FP16='FP16'
DEVICE=$DEVICE
NUM_REQS=2
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"

IR_FP16="$XML_IR_FP16/mobilenet-ssd.xml"

Sample_name="safety_gear_detection.py"

SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution


#source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh

python3 -V

if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder \$FP16"
   mkdir -p $Output_folder_16 
   mkdir -p $XML_IR_FP16 
   mo \
   --input_model resources/worker_safety_mobilenet.caffemodel \
   --model_name mobilenet-ssd \
   --data_type $FP16 \
   --output_dir $XML_IR_FP16 \

   python3 $Sample_name  -i $INPUT_FILE  -m  $IR_FP16  --labels labels.txt -o $Output_folder_16 -d $DEVICE -nireq $NUM_REQS


   #source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh
   python3 safety_gear_detection_annotate.py -i $INPUT_FILE \
                                     -o $Output_folder_16 \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION
   
   #source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh
fi



