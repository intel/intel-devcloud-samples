# alias python3=python3.6

python3.6 -V

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


source /opt/intel/openvino/bin/setupvars.sh

DEVICE=$device
INPUT_FILE="reference_files/Safety_Full_Hat_and_Vest.mp4"
NUM_REQS=2
LABEL_FILE="reference_files/labels.txt"


if [[ ! -z "$FP16" ]];
then
	# Make sure that the output directory exists.
	OUTPUT_FILE_FP16="data/output_safety_gear_detection/FP16/"
	# Set inference model IR files using specified precision
	MODELPATH_FP16="models/mobilenet-ssd/FP16/mobilenet-ssd.xml"
	echo "Creating output folder \$FP16"
	mkdir -p $OUTPUT_FILE_FP16
	mkdir -p models/mobilenet-ssd/FP16/


	# Create FP16 IR files
	python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
	--input_model reference_files/worker_safety_mobilenet.caffemodel \
	--model_name mobilenet-ssd \
	--data_type FP16 \
	--output_dir models/mobilenet-ssd/FP16 \

	# Store input arguments: <output_directory> <device> <input_file> <num_reqs>
	
	# Run the safety gear detection code - FP16 model
	python3 safety_gear_detection.py -m $MODELPATH_FP16 \
									 -i $INPUT_FILE \
									 -o $OUTPUT_FILE_FP16 \
									 -d $DEVICE \
									 -nireq $NUM_REQS \
									 --labels $LABEL_FILE


	# Run the output video annotator code
	SCALE_FRAME_RATE=1    # scale number or output frames to input frames
	SCALE_RESOLUTION=0.5  # scale output frame resolution 
	python3 safety_gear_detection_annotate.py -i $INPUT_FILE \
											  -o $OUTPUT_FILE_FP16 \
											  -f $SCALE_FRAME_RATE \
											  -s $SCALE_RESOLUTION

fi

if [[ ! -z "$FP32" ]];
then
	OUTPUT_FILE_FP32="data/output_safety_gear_detection/FP32/"
	# Set inference model IR files using specified precision
	MODELPATH_FP32="models/mobilenet-ssd/FP32/worker_safety_mobilenet.xml"
	echo "Creating output folder \$FP32"
	mkdir -p $OUTPUT_FILE_FP32
	mkdir -p models/mobilenet-ssd/FP32/

	# Create FP32 IR files
	python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
	--input_model reference_files/worker_safety_mobilenet.caffemodel \
	--data_type FP32 \
	--output_dir models/mobilenet-ssd/FP32 \

	
	# Store input arguments: <output_directory> <device> <input_file> <num_reqs>
	
	# Run the safety gear detection code - FP32 model
	python3 safety_gear_detection.py -m $MODELPATH_FP32 \
									 -i $INPUT_FILE \
									 -o $OUTPUT_FILE_FP32 \
									 -d $DEVICE \
									 -nireq $NUM_REQS \
									 --labels $LABEL_FILE


	# Run the output video annotator code
	SCALE_FRAME_RATE=1    # scale number or output frames to input frames
	SCALE_RESOLUTION=0.5  # scale output frame resolution 
	python3 safety_gear_detection_annotate.py -i $INPUT_FILE \
											  -o $OUTPUT_FILE_FP32 \
											  -f $SCALE_FRAME_RATE \
											  -s $SCALE_RESOLUTION

fi