echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "INPUT_FILE:  $INPUT_FILE"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "Running on" $RUN_ON_PREM

INPUT_FOLDER="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/yolo-v8/images"
Output_folder="$RUN_ON_PREM/$OUTPUT_FOLDER/results/"

source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh

echo "Creating output folder"
mkdir -p $Output_folder

# Run the YOLO V8 object detection code - FP32 model
python3 object_detection_demo_yolov8.py -i $INPUT_FOLDER -o $Output_folder


