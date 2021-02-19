cd $PBS_O_WORKDIR

mkdir -p logs/
saved_model_dir=$(find ./tf_saved_model/ -maxdepth 1 -type d | tail -n 1)

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
    --model_name "biobert"                                            \
    --saved_model_dir $saved_model_dir/                               \
    --output_dir ./ov/                                                \
    --disable_nhwc_to_nchw                                            \
    --input "input_ids,segment_ids,input_mask"                        \
    --output "unstack"                                                \
    --data_type FP16                                                  \
    --batch 1
