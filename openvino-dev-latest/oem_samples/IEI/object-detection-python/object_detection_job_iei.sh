
# Store input arguments: <output_directory> <device> <fp_precision> <input_file> <num_reqs>
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4
NUM_REQS=$5

# The default path for the job is the user's home directory,
#  change directory to where the files are.
cd $PBS_O_WORKDIR

# Make sure that the output directory exists.
mkdir -p $OUTPUT_FILE

# Set inference model IR files using specified precision
MODELPATH=models/mobilenet-ssd/${FP_MODEL}/mobilenet-ssd.xml
LABEL_FILE=labels.txt

# Run the object detection code
python3 object_detection.py -m $MODELPATH \
                            -i $INPUT_FILE \
                            -o $OUTPUT_FILE \
                            -d $DEVICE \
                            -nireq $NUM_REQS \
                            --labels $LABEL_FILE

# Run the output video annotator code
SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution 
python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $OUTPUT_FILE \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION
