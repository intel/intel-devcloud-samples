python3 /opt/intel/openvino_2021.2.185/deployment_tools/tools/model_downloader/downloader.py  --name mobilenet-ssd -o raw_models 

mkdir -p models/mobilenet-ssd/FP16
mkdir -p models/mobilenet-ssd/FP32


# Create FP16 IR files
python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py \
--input_model raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
--data_type FP16 \
--output_dir models/mobilenet-ssd/FP16 \
--scale 256 \
--mean_values [127,127,127]

# Create FP32 IR files
python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py \
--input_model raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
--data_type FP32 \
--output_dir models/mobilenet-ssd/FP32 \
--scale 256 \
--mean_values [127,127,127]


