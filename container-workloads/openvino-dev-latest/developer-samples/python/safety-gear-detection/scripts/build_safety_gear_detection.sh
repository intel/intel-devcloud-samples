sample_name="safety_gear_detection_2022.1"
docker_filename="openvino_cgvh_dev_2022.1.dockerfile"
sudo docker build -t $sample_name -f ./developer-samples/python/safety-gear-detection/dockerfile/ubuntu18/$docker_filename  .

