### Intel devcloud sample containerization  

This example introduces the containerized object detection using a pre-trained mobilenet deep learning model  to detect vehicles 
with qrpo,applicationwriter library calls. 
This example demonstrates key concepts of OpenVINO Latest version: 2021.3 and OpenVINO LTS version: 2020.3.2 to do inferencing on Intel® Core™ CPUs.

### HW and SW prerequistes for development and running : Ubuntu18 machine with latest version of  docker installed 

### Steps to build and run the containerized Object detection sample with  OpenVino docker file.
### Checkout the repo: https://gitlab.devtools.intel.com/iot-devcloud/reference-samples

1. Build the Ubuntu18 Openvino-lts plus the object detection sample docker image  using the dockerfile in dockerfile/Ubuntu18 folder
Each sample has its own dockerfile and  a config file which allows users to configure and build the dockerfile with custome requirements of 
device (CPU,GPU,VPU etc), Precision(FP32, FP16) and Openvino Package Version(2021.3.394 etc) 

### GoTo the level container_samples/openvino-lts in the repo
Execute the command :
    sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2020.3.2/l_openvino_toolkit_dev_ubuntu18_p_2020.3.355.tgz 
	--build-arg GMMLIB=19.3.2 
	--build-arg IGC_CORE=1.0.2597 
	--build-arg IGC_OPENCL=1.0.2597 
	--build-arg INTEL_OPENCL=19.41.14441 
	--build-arg INTEL_OCLOC=19.41.14441
	--build-arg device="CPU"
	-t object_detection_2020lts:latest 
	-f ./object-detection-python/dockerfile/ubuntu18/openvino_cgvh_dev_2020.3.2.dockerfile .

2. Running the container without Jupyterhub:
      sudo docker run -it object_detection_2020lts:latest
	  
### Steps to build and run  the containerized Object detection sample with OpenVino Latest version 
###  GoTo the level container_samples/openvino-latest  in the repo
1. Build the Ubuntu18 Openvino-latest plus the object detection sample docker image  using the dockerfile in dockerfile/Ubuntu18 folder
Execute the command :
sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.3/l_openvino_toolkit_dev_ubuntu18_p_2021.3.394.tgz 
--build-arg GMMLIB=19.3.2 
--build-arg IGC_CORE=1.0.2597 
--build-arg IGC_OPENCL=1.0.2597
 --build-arg INTEL_OPENCL=19.41.14441
 --build-arg INTEL_OCLOC=19.41.14441 
 --build-arg device="CPU" 
 -t object_det_2021.3:latest 
 -f ./object-detection-python/dockerfile/ubuntu18/openvino_cgvh_dev_2021.3.dockerfile .

2. Running the container :
      sudo docker run -it object_det_2021.3:latest

### Note:  To run any other container besides object_detection, change the name of the container appropriately for lts and latest versions of OpenVINO 