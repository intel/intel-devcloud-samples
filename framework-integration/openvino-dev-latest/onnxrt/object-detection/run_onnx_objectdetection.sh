#!/usr/bin/env python3

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "MODEL:  $MODEL"
export INPUT_FILE="/object-detection-onnxrt/data/"$INPUT_FILE
echo "Input file $INPUT_FILE"

export MODEL="/object-detection-onnxrt/data/"$MODEL
source /opt/intel/openvino_2022/setupvars.sh

echo "Using Openvino Integration with ONNXRT"
python3 ONNX_object_detection.py -m $MODEL -i $INPUT_FILE -d $DEVICE
