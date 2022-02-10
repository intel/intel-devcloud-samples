#!/usr/bin/env python3.7

echo "Openvino Package Version: openvino_$OPENVINO_VERSION"
echo "Device:  $DEVICE"
echo "MODEL:  $MODEL"
echo "Input image" $INPUT_FILE


export MODEL="/classification-ovtf/data/"$MODEL
echo $MODEL >> /mount_folder/model.txt
cp -ir $INPUT_FILE /mount_folder/
export LABELS="/classification-ovtf/data/"$LABELS


source /opt/intel/openvino/bin/setupvars.sh

echo "Using Openvino Integration with Tensorflow"

export FLAG="openvino"


python3 classification_sample_video_image.py -f $FLAG | tee /mount_folder/result_ovtf.txt


echo "Using Stock Tensorflow"

export FLAG="native"

python3 classification_sample_video_image.py -f $FLAG | tee -a  /mount_folder/result_tf.txt


sleep 10

