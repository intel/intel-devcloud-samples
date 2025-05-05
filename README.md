# PROJECT NOT UNDER ACTIVE MANAGEMENT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
###  Intel® DevCloud Containerized Reference Samples 

The Intel® DevCloud containerized **marketplace** reference samples enables users to seamlessly build and test containerized AI inference workloads on Intel® hardware specialized for deep learning. The containerized refrence samples contain optimized deep-learning models pre-built with the Intel® Distribution of OpenVINO™ toolkit to do the inferencing on Intel® Core™ CPUs i3, i5, i7 and Xeons.

Each sample contains instructions for:
* How It Works? 
* Supported runtime customizations
* Building and running on Intel® DevCloud and your local system

### OpenVINO™ Samples 

Container applications demonstrating inference pipelines with Intel® Distribution of OpenVINO™ toolkit - Inference Engine. 

[![Stable release](https://img.shields.io/badge/version-2021.4.2-blue.svg)](https://github.com/openvinotoolkit/openvino/releases/tag/2021.4.2) 

| Application | Description |
| --- | --- |
| [Safety Gear Detection](container-workloads/openvino-dev-latest/developer-samples/python/safety-gear-detection/README.md) | Use an optimized and pre-trained MobileNet-SSD neural network to detect people and their safety gear from video input. |
| [People Counter System](container-workloads/openvino-dev-latest/developer-samples/python/people-counter/README.md) | Deploy a smart video IoT solution using a person detection model from Intel® Distribution of OpenVINO™ toolkit to detect and counter people in each frame of a video feed. |
| [Accelerated Object Detection](https://github.com/intel-innersource/containers.docker.devcloud.reference-samples/blob/readme-updates/container-workloads/openvino-dev-latest/developer-samples/python/object-detection/README.md) | Accelerate object detection by using asynchronous inferencing and distributing workloads to multiple types of processing units. |
| [Tiny YOLO V3 Object Detection](container-workloads/openvino-dev-latest/developer-samples/python/tiny-yolo-v3/README.md) | Convert a pre-trained DarkNet YOLO V3 model to TensorFLow, then run accelerated inference using OpenVINO™ for object detection. Learn how to fine-tune an application for optimal performance. |
| [Benchmark Sample](container-workloads/openvino-dev-latest/developer-samples/python/benchmark/README.md) | Learn how to use the Intel® DevCloud benchmarking tool to evaluate the performance of your model's synchronous and asynchronous inference. |
| [Deep Learning Streamer](container-workloads/openvino-dev-latest/tutorials/python/dlstreamer/README.md) | Learn how to utilize the GStreamer* plug-in to manage complex media analytics pipelines and boost your AI inferencing capabilities. |
| [Pneumonia Classification](container-workloads/openvino-dev-latest/developer-samples/python/pneumonia-classification/README.md) | Classify the probability of pneumonia in X-Ray images using a pre-trained neural network and the Intel® Distribution of OpenVINO™ toolkit.|

### OpenVINO™ Integration with TensorFlow Samples

TensorFlow* container applications with OpenVINO™ toolkit optimizations.

[![Stable release](https://img.shields.io/badge/version-v1.1.0-blue.svg)](https://github.com/openvinotoolkit/openvino_tensorflow/releases/tag/v1.1.0) 

| Application | Description |
| --- | --- |
| [Object Detection](framework-integration/openvino-dev-latest/openvino-tensorflow/object-detection/README.md) | The sample showcases object detection using YoloV3 TensorFlow Model on OpenVINO™ integration with Tensorflow. |
| [Classification](framework-integration/openvino-dev-latest/openvino-tensorflow/classification/README.md) | The sample is to showcase classification of image with inception V3 Tensorflow model using OpenVINO™ integration with Tensorflow. |

### OpenVINO™ Integration with Torch-ORT Samples

PyTorch* container applications with OpenVINO™ toolkit optimizations.

[![Stable release](https://img.shields.io/badge/version-1.13.1-blue.svg)](https://github.com/pytorch/ort)

| Application | Description |
| --- | --- |
| [Image Classification](framework-integration/openvino-dev-latest/openvino-torch-ort/image-classification/README.md) | The sample showcases image classification using ResNet-50 PyTorch Model on OpenVINO™ integration with Torch-ORT. |
| [Sequence Classification](framework-integration/openvino-dev-latest/openvino-torch-ort/sequence-classification/README.md) | The sample is to showcase sequence classification of text with BERT PyTorch model using OpenVINO™ integration with Torch-ORT. |
