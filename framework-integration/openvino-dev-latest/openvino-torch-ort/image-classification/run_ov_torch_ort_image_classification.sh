echo "Provider:  $PROVIDER"
echo "Backend: $BACKEND"
echo "Precision: $PRECISION"
echo "Input image: $INPUT_FILE"

mkdir -p /mount_folder
cp -ir $INPUT_FILE /mount_folder/

echo "Using Openvino Integration with Torch-ORT"
python3 resnet_image_classification.py \
        --input-file $INPUT_FILE \
        --labels $LABELS \
        --provider $PROVIDER \
        --backend $BACKEND \
        --precision $PRECISION \
        | tee /mount_folder/result_ov_torch_ort.txt

echo "Using Stock Pytorch"
python3 resnet_image_classification.py --pytorch-only \
        --input-file $INPUT_FILE \
        --labels $LABELS \
        | tee /mount_folder/result_stock_pytorch.txt

sleep 10