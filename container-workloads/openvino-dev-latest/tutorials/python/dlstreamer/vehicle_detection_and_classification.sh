#!/bin/bash

DETECT_MODEL="$RUN_ON_PREM/$OUTPUT_FOLDER/models/intel/pedestrian-and-vehicle-detector-adas-0001/FP16/pedestrian-and-vehicle-detector-adas-0001.xml"
CLASSIFY_MODEL="$RUN_ON_PREM/$OUTPUT_FOLDER/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml"
CLASSIFY_MODEL_PROC="model_proc/vehicle-attributes-recognition-barrier-0039.json"
INFERENCE_INTERVAL=1
OUTPUT_FILE="$RUN_ON_PREM/$OUTPUT_FOLDER/$DEVICE/vehicle_detect_classify.mp4"
ENCODING_TYPE="avenc_mpeg4"

PIPELINE="filesrc location=$INPUT_FILE ! decodebin ! \
    videoconvert n-threads=4 ! capsfilter caps=\"video/x-raw,format=BGRx\" ! \
    gvadetect model=$DETECT_MODEL device=$DEVICE inference-interval=$INFERENCE_INTERVAL ! \
    gvaclassify model=$CLASSIFY_MODEL model-proc=$CLASSIFY_MODEL_PROC device=$DEVICE ! \
    gvawatermark ! videoconvert ! videoscale ! video/x-raw,width=640,height=360 ! $ENCODING_TYPE ! mp4mux ! filesink location=$OUTPUT_FILE"
echo "gst-launch-1.0 $PIPELINE"

mkdir -p $(dirname "${OUTPUT_FILE}")
rm -f "$OUTPUT_FILE"
gst-launch-1.0 $PIPELINE
