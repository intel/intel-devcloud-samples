mkdir -p  /opt/intel/openvino_2021.2.185/python/samples/models/retinanet-tf/FP16
python3 /opt/intel/openvino_2021.2.185/deployment_tools/tools/model_downloader/downloader.py  --name retinanet-tf -o raw_models 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model raw_models/public/retinanet-tf/retinanet_resnet50_coco_best_v2.1.0.pb \
--input_shape [1,1333,1333,3] \
--input input_1 \
--mean_values [103.939,116.779,123.68] \
--output filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 \
--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/retinanet.json \
--output_dir /opt/intel/openvino_2021.2.185/python/samples/models/retinanet-tf/FP16 \
--data_type FP16 \
--model_name retinanet-tf
python3 /opt/intel/openvino_2021.2.185/python/samples/object_detection.py -i cars_1900_first_frame.jpg  -m  /opt/intel/openvino_2021.2.185/python/samples/models/retinanet-tf/FP16/retinanet-tf.xml  --labels labels.txt -o ./

