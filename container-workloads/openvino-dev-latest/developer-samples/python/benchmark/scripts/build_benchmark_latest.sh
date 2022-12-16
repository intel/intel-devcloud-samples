tag=2022.2
sudo docker build --no-cache  -t benchmarking_$tag:latest -f ./developer-samples/python/benchmark/dockerfile/ubuntu18/openvino_cgvh_dev_$tag.dockerfile .
