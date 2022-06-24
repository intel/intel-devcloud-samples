#!/bin/bash

NUM_BUFFERS=-1
DETECT_MODEL="$RUN_ON_PREM/$OUTPUT_FOLDER/models/intel/pedestrian-and-vehicle-detector-adas-0001/FP16/pedestrian-and-vehicle-detector-adas-0001.xml"
CLASSIFY_MODEL="$RUN_ON_PREM/$OUTPUT_FOLDER/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml"
CLASSIFY_MODEL_PROC="model_proc/vehicle-attributes-recognition-barrier-0039.json"
OUTPUT_FILE="$RUN_ON_PREM/$OUTPUT_FOLDER/$DEVICE/log_detect_track_classify.txt"
INFERENCE_INTERVAL=10
CHANNELS_COUNT=4
PIPELINE="filesrc location=$INPUT_FILE num-buffers=$NUM_BUFFERS ! decodebin ! \
    videoconvert n-threads=4 ! capsfilter caps="video/x-raw,format=BGRx" ! \
    gvadetect model=$DETECT_MODEL device=$DEVICE inference-interval=$INFERENCE_INTERVAL ! queue ! \
    gvatrack ! queue ! gvaclassify model=$CLASSIFY_MODEL model-proc=$CLASSIFY_MODEL_PROC device=$DEVICE reclassify-interval=$INFERENCE_INTERVAL ! queue ! \
    gvafpscounter starting-frame=10 ! fakesink "
FINAL_PIPELINE_STR=""
for (( CURRENT_CHANNELS_COUNT=0; CURRENT_CHANNELS_COUNT < $CHANNELS_COUNT; ++CURRENT_CHANNELS_COUNT ))
do
  FINAL_PIPELINE_STR+=$PIPELINE
done
echo "gst-launch-1.0 $FINAL_PIPELINE_STR"

mkdir -p $(dirname "${OUTPUT_FILE}")
rm -f "$OUTPUT_FILE"
gst-launch-1.0 $FINAL_PIPELINE_STR | tee $OUTPUT_FILE
