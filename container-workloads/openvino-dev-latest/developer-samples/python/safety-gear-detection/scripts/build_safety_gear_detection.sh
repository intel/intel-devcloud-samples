sample_name="safety_gear_detection_2023.0.0"
docker_filename="openvino_cgvh_dev_2023.0.0.dockerfile"
sudo docker build -t $sample_name -f ./developer-samples/python/safety-gear-detection/dockerfile/ubuntu20/$docker_filename  .

