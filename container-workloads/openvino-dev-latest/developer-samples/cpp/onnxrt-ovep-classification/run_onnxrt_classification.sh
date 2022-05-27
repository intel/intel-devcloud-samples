echo "Device:  $DEVICE"
echo "MODEL:  $MODEL"
echo "Input image: $INPUT_FILE"

mkdir -p /mount_folder
export MODEL="/classification-onnxrt/data/"$MODEL
export INPUT_FILE="/classification-onnxrt/data/"$INPUT_FILE
cp -r $INPUT_FILE /mount_folder/
export LABELS="/classification-onnxrt/data/"$LABELS


source /opt/intel/openvino_2022/setupvars.sh

echo "Using Openvino Integration with ONNXRT"

sleep 5
mkdir build
cd build
cmake ..
make -j4
./run_squeezenet $EXECUTION_PROVIDER $DEVICE $MODEL $INPUT_FILE $LABELS

