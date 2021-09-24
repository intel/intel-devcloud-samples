#!/bin/bash

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Output Folder:  $OUTPUT_FOLDER"
echo "Running on: $RUN_ON_PREM"
echo "Input file: $INPUT_FILE"

echo "Creating output directory $RUN_ON_PREM/$OUTPUT_FOLDER/$DEVICE"
mkdir -p "$RUN_ON_PREM/$OUTPUT_FOLDER/$DEVICE"

if [ "$DEVICE" == "CPU" ]
then
   echo "CPU supports FP16, internally upscaling to FP32. Preferred usage for this tutorial as iGPU and VPU support only FP16."
fi

echo "Downloading pedestrian-and-vehicle-detector and vehicle-attributes-recognition FP16 models"
python3 ${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py \
         --name pedestrian-and-vehicle-detector-adas-0001 \
         --precisions FP16 \
         -o "$RUN_ON_PREM/$OUTPUT_FOLDER/models"

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py \
         --name vehicle-attributes-recognition-barrier-0039 \
         --precisions FP16 \
         -o "$RUN_ON_PREM/$OUTPUT_FOLDER/models"


source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh

source vehicle_detection_and_classification.sh

source vehicle_detection_benchmark.sh

source vehicle_tracking_benchmark.sh



