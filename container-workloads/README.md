### Intel devcloud sample containerization  with qrpo, application writer libraies

This example introduces the containerized object detection using a pre-trained mobilenet deep learning model  to detect vehicles with qrpo,applicationwriter library calls. This example demonstrates key concepts of OpenVINO 2021.2, to do  the inferencing on Intel® Core™ CPUs.

Following are the steps and commands to build and run the devcloud container.

Checkout the repo : https://github.com/intel-innersource/containers.docker.devcloud.reference-samples/tree/object_detection_2021.4_new/container-workloads/

Steps to run containerized Object detection sample using  the prebuild Ubuntu 18, OpenVINO2021.4 along with third party libraries of qarpo, application writer
Image located at : quay.io/devcloud/devcloud-openvino-data-dev:2021.4_latest


1. Build  the object detection sample docker image  along with the prebuild image of OpenVINO and devcloud third party utils of application writer and qrpo  using the dockerfile located in container-workloads/openvino-dev-latest/developer-samples/python/object-detection/dockerfile/ubuntu18 folder of the repo:
   

1. Goto the level container-workloads/openvino-dev-latest in the repo:
2. sudo docker build -t object_detection_2021.4 -f ./developer-samples/python/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile  .



2.Running the container:
     sudo docker run -it object_detection_2021.4:latest

3.Running the container with user defined environment variables of PRECISION, MODEL, INPUT_FILE, RUN_ON_PREM
     sudo docker run  --env-file conftest.env object_detection_2021.4:latest

 where conftest.env
    
      DEVICE=CPU
      PRECISION=FP32,FP16
      OUTPUT_FOLDER=output_benchmark_latest
      RUN_ON_PREM=data
      INPUT_FILE="suv.mp4"
      MODEL=Pytorch


Note:  To run any other container besides object_detection, change the name of the container appropriately