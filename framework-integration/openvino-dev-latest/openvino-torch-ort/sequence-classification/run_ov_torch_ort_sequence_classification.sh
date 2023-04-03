echo "Provider:  $PROVIDER"
echo "Backend: $BACKEND"
echo "Precision: $PRECISION"
if [ -z "$INPUT_FILE" ];
then
    echo "Input sentence: $INPUT"
else
    echo "Input file: $INPUT_FILE"
fi

mkdir -p /mount_folder
cp -ir $INPUT_FILE /mount_folder/

echo "Using Openvino Integration with Torch-ORT"
if [ ! -z "$INPUT_FILE" ]; then
    python3 bert_for_sequence_classification.py \
        --input-file $INPUT_FILE \
        --provider $PROVIDER \
        --backend $BACKEND \
        --precision $PRECISION \
        | tee /mount_folder/result_ov_torch_ort.txt
else
    python3 bert_for_sequence_classification.py \
    --input "$INPUT" \
    --provider $PROVIDER \
    --backend $BACKEND \
    --precision $PRECISION \
    | tee /mount_folder/result_ov_torch_ort.txt
fi

echo "Using Stock Pytorch"
if [ ! -z "$INPUT_FILE" ]; then
    python3 bert_for_sequence_classification.py --pytorch-only \
        --input-file $INPUT_FILE \
        | tee /mount_folder/result_stock_pytorch.txt
else
    python3 bert_for_sequence_classification.py --pytorch-only \
        --input "$INPUT" \
        | tee /mount_folder/result_stock_pytorch.txt
fi

sleep 10