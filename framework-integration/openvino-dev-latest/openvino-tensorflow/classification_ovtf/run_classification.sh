#!/usr/bin/env python3.7

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "MODEL:  $MODEL"
echo "Input image" $INPUT_FILE


export MODEL="/OVTF/classification_ovtf/data/"$MODEL
echo $MODEL >> /mount_folder/result_model.txt
export LABELS="/OVTF/classification_ovtf/data/"$LABELS

ls -l /OVTF/classification_ovtf/data/

source /opt/intel/openvino/bin/setupvars.sh

python3 classification_sample_video_image.py -m $MODEL -i $INPUT_LAYER -o $OUTPUT_LAYER -ip $INPUT_FILE -l $LABELS -it $INPUT_TYPE -d $DEVICE -f $FLAG | tee /mount_folder/result_infer_ovtf.txt
