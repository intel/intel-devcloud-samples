# alias python3=python3.6

python3.6 -V

set -e

#check if config file is passed as an argument

if [[ -z "$1" ]];then
	echo "Config file is missing!"
	exit 1
fi

#echo "$(cat $1)"
device=$(cut -d "," -f 1  $1 |xargs)
FP16=$(cut -d "," -f 2  $1 | xargs) 
FP32=$(cut -d "," -f 3  $1 | xargs)
ver=$(cut -d "," -f 4  $1 | xargs)
openvino_ver=openvino_$ver

echo "Openvino Package Version: $openvino_ver"
echo "Device:  $device"
echo "Precision:  $FP16, $FP32"

# Store input arguments: <output_directory> <device> <input_file> <num_reqs>


DEVICE=$device
INPUT_FILE="classroom.mp4"
NUMREQUEST=2
THRESHOLD=0.4
NUMSTREAMS=1
LABEL_FILE="coco.names"
source /opt/intel/openvino/bin/setupvars.sh

# Copy some prerequisits
cp -rf /opt/intel/openvino/deployment_tools/inference_engine/demos/common/ .
cp common/python/monitors.py .

#Download Tiny YOLO V3 Darknet Model Weights and COCO labels file
curl https://pjreddie.com/media/files/yolov3-tiny.weights > yolov3-tiny.weights
curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names> coco.names

#clone the tensorflow-yolo-v3 repository to access the convert_weights_pb.py python script that can convert all different types of YOLO and Tiny YOLO models to frozen Tensorflow Protobuf files (.pb)
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
python3.6 tensorflow-yolo-v3/convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny 



if [[ ! -z "$FP16" ]];
then
	# Set inference model IR files using specified precision
	OUTPUT_FILE_FP16="data/output_tiny_yolo_v3/FP16/"
	MODELPATH_FP16="models/tinyyolov3/FP16/frozen_darknet_yolov3_model.xml"
	mkdir -p models/tinyyolov3/FP16/

	#Create the IR files for the inference model - FP16
	python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
	--input_model frozen_darknet_yolov3_model.pb \
	--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json \
	--data_type FP16 \
	--batch 1 \
	--output_dir models/tinyyolov3/FP16

	# Make sure that the output directory exists.
	mkdir -p $OUTPUT_FILE_FP16

	# Run the Tiny YOLO V3 object detection code - FP16 model
	python3 object_detection_demo_yolov3_async.py -m $MODELPATH_FP16 \
		-i $INPUT_FILE \
		-o $OUTPUT_FILE_FP16 \
		-d $DEVICE \
		-t $THRESHOLD \
		-nireq $NUMREQUEST \
		-nstreams $NUMSTREAMS \
		-no_show \
		--labels $LABEL_FILE 
fi


if [[ ! -z "$FP32" ]];
then
	# Set inference model IR files using specified precision
	OUTPUT_FILE_FP32="data/output_tiny_yolo_v3/FP32/"
	MODELPATH_FP32="models/tinyyolov3/FP32/frozen_darknet_yolov3_model.xml"

	mkdir -p models/tinyyolov3/FP32/
	#Create the IR files for the inference model - FP32
	python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
	--input_model frozen_darknet_yolov3_model.pb \
	--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json \
	--data_type FP32 \
	--batch 1 \
	--output_dir models/tinyyolov3/FP32


	# Make sure that the output directory exists.
	mkdir -p $OUTPUT_FILE_FP32

	# Run the Tiny YOLO V3 object detection code - FP32 model
	python3 object_detection_demo_yolov3_async.py -m $MODELPATH_FP32 \
		-i $INPUT_FILE \
		-o $OUTPUT_FILE_FP32 \
		-d $DEVICE \
		-t $THRESHOLD \
		-nireq $NUMREQUEST \
		-nstreams $NUMSTREAMS \
		-no_show \
		--labels $LABEL_FILE 
fi