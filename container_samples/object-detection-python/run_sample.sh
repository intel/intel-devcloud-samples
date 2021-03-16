#!/usr/bin/env bash 
INPUT_FILE="/opt/intel/openvino_2021.2.185/python/samples/object-detection-python/cars_1900.mp4"
DEVICE="CPU"
NUM_REQS=2
Output_folder="output"
XML_IR_FP16="models/mobilenet-ssd/FP16/mobilenet-ssd.xml"
XML_IR_FP32="models/mobilenet-ssd/FP32/mobilenet-ssd.xml"
Sample_name="object_detection.py"

source /opt/intel/openvino_2021.2.185/bin/setupvars.sh

alias python3=python3.6
echo $Output_folder
echo $XML_IR_FP16
echo $XML_IR_FP32
echo $INPUT_FILE
echo $Sample_name

mkdir -p $Output_folder 

python3 $Sample_name  -i $INPUT_FILE  -m  $XML_IR_FP16  --labels labels.txt -o $Output_folder -d $DEVICE -nireq $NUM_REQS
python3 $Sample_name  -i $INPUT_FILE  -m  $XML_IR_FP32  --labels labels.txt -o $Output_folder -d $DEVICE -nireq $NUM_REQS

# Run the output video annotator code
SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution
python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $Output_folder \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION
