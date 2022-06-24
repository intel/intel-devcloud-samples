echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device: $DEVICE"
echo "Precision: $PRECISION"
echo "OUTPUT FOLDER: $OUTPUT_FOLDER"
echo "MODEL: $MODEL"
echo "Running on: $RUN_ON_PREM"
echo "Input image: $INPUT_FILE"


FP16='FP16'
DEVICE=$DEVICE
NUM_REQS=2

Output_folder_16="$RUN_ON_PREM/$OUTPUT_FOLDER/results/FP16"
mkdir -p $Output_folder_16

Sample_name="people_counter.py"

source /opt/intel/openvino_$OPENVINO_VERSION/setupvars.sh


omz_downloader --name $MODEL -o $RUN_ON_PREM/$OUTPUT_FOLDER/ir_models 

# The following variables are assigned to the path of the optimized IR version of the
# models downloaded from "intel" directory of Intel Open Model Zoo repository. If other models
# are used, these variables should accordingly be assigned to the path of the IR version of
# those other models.

IR_FP16=$RUN_ON_PREM/$OUTPUT_FOLDER/ir_models/intel/$MODEL/$FP16/$MODEL.xml


if [[ "$PRECISION" == *"$FP16"* ]];
then
   echo "Creating output folder $Output_folder_16"
   mkdir -p $Output_folder_16 
   python3 $Sample_name  -i $INPUT_FILE \
                         -m $IR_FP16 \
                         -o $Output_folder_16 \
                         -d $DEVICE \
                         -nir $NUM_REQS \
                         -pt 0.7
fi
