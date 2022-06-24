tag=2022.1
sudo docker build -t pneumonia_classification_$tag:latest -f ./developer-samples/python/pneumonia-classification/dockerfile/ubuntu18/openvino_cgvh_dev_$tag.dockerfile  .
