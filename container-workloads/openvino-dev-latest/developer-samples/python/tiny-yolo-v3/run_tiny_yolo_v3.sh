# alias python3=python3.6

#python3.6 -V

#set -e

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "Precision:  $PRECISION"
echo "INPUT_FILE:  $INPUT_FILE"
echo "OUTPUT FOLDER:  $OUTPUT_FOLDER"
echo "Running on" $RUN_ON_PREM


FP16='FP16'
FP32='FP32'
INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/tiny-yolo-v3/classroom.mp4"
DEVICE=$DEVICE
NUM_REQS=2
API="async"
Output_folder_32="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP32"
Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
XML_IR_FP16="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP16"
XML_IR_FP32="$RUN_ON_PREM/$OUTPUT_FOLDER/IR/FP32"

#Store input arguments: <output_directory> <device> <input_file> <num_reqs>


#INPUT_FILE="classroom.mp4"
NUMREQUEST=2
THRESHOLD=0.4
NUMSTREAMS=1
LABEL_FILE="coco.names"
source /opt/intel/openvino_$OPENVINO_VERSION/bin/setupvars.sh


#Download Tiny YOLO V3 Darknet Model Weights and COCO labels file
curl https://pjreddie.com/media/files/yolov3-tiny.weights > yolov3-tiny.weights
curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names> coco.names

#clone the tensorflow-yolo-v3 repository to access the convert_weights_pb.py python script that can convert all different types of YOLO and Tiny YOLO models to frozen Tensorflow Protobuf files (.pb)
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
python3.6 tensorflow-yolo-v3/convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny 



if [[ "$PRECISION" == *"$FP16"* ]];
then
        echo "Creating output folder \$FP16"
        mkdir -p $Output_folder_16
        mkdir -p $XML_IR_FP16
	#Create the IR files for the inference model - FP16
	python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
	--input_model frozen_darknet_yolov3_model.pb \
	--transformations_config /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json \
	--data_type $FP16 \
	--batch 1 \
	--output_dir $XML_IR_FP16


	# Run the Tiny YOLO V3 object detection code - FP16 model
	python3 object_detection_demo_yolov3_async.py -m $XML_IR_FP16/frozen_darknet_yolov3_model.xml \
		-i $INPUT_FILE \
		-o $Output_folder_16 \
		-d $DEVICE \
		-t $THRESHOLD \
		-nireq $NUMREQUEST \
		-nstreams $NUMSTREAMS \
		-no_show \
		--labels $LABEL_FILE 
fi


if [[ "$PRECISION" == *"$F32"* ]];
then

        echo "Creating output folder \$FP32"
        mkdir -p $Output_folder_32
        mkdir -p $XML_IR_FP32
	#Create the IR files for the inference model - FP32
	python3 /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/mo.py \
	--input_model frozen_darknet_yolov3_model.pb \
	--transformations_config /opt/intel/openvino_$OPENVINO_VERSION/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json \
	--data_type $FP32 \
	--batch 1 \
	--output_dir $XML_IR_FP32



        echo $XML_IR_FP32
         # Run the Tiny YOLO V3 object detection code - FP16 model
        python3 object_detection_demo_yolov3_async.py -m $XML_IR_FP32/frozen_darknet_yolov3_model.xml \
                 -i $INPUT_FILE \
                 -o $Output_folder_32 \
                 -d $DEVICE \
                 -t $THRESHOLD \
                 -nireq $NUMREQUEST \
                 -nstreams $NUMSTREAMS \
                 -no_show \
                 --labels $LABEL_FILE

fi
