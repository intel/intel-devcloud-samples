#!/usr/bin/env python3.7

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "MODEL:  $MODEL"
echo "Input image" $INPUT_FILE


export MODEL="/object-detection-ovtf/data/"$MODEL
#echo $MODEL >> /mount_folder/result_model.txt
export LABELS="/object-detection-ovtf/data/"$LABELS


source /opt/intel/openvino/bin/setupvars.sh


echo "Using Openvino Integration with Tensorflow"
export OUTPUT_FILENAMES="ovtf_"$OUTPUT_FILENAME
python3 object_detection_sample_video_image.py -m $MODEL -i $INPUT_LAYER -o $OUTPUT_LAYER -ip $INPUT_FILE -l $LABELS -it $INPUT_TYPE -d $DEVICE -f $FLAG -ih $INPUT_HEIGHT -iw $INPUT_WIDTH -of $OUTPUT_FILENAMES --output_dir $OUTPUT_DIRECTORY | tee /mount_folder/result_ovtf.txt

echo "Using Stock Tensorflow"

export FLAG="native"
export OUTPUT_FILENAMES="tf_"$OUTPUT_FILENAME
python3 object_detection_sample_video_image.py -m $MODEL -i $INPUT_LAYER -o $OUTPUT_LAYER -ip $INPUT_FILE -l $LABELS -it $INPUT_TYPE -d $DEVICE -f $FLAG -ih $INPUT_HEIGHT -iw $INPUT_WIDTH -of $OUTPUT_FILENAMES --output_dir $OUTPUT_DIRECTORY | tee /mount_folder/result_tf.txt

sleep 10
