### Object Detection Container without qrpo, application writer job id libraies

This example introduces the containerized object detection using a pre-trained deep learning model  to detect vehicles without qrpo,applicationwriter and jobid library calls. This example demonstrates key concepts of OpenVINO, to do  the inferencing on Intel® Core™ CPUs

Steps to run containerized Object detection sample

1. Build the Ubuntu Openvino plus the object detection sample docker image  using the dockerfile/Ubuntu18_OpenVino folder  :
     sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.2/l_openvino_toolkit_dev_ubuntu18_p_2021.2.185.tgz  --build-arg GMMLIB=19.3.2  --build-arg IGC_CORE=1.0.2597  --build-arg IGC_OPENCL=1.0.2597  --build-arg INTEL_OPENCL=19.41.14441  --build-arg INTEL_OCLOC=19.41.14441  -t ubuntu18_dev:2021.2 -f /dockerfiles/ubuntu18-OpenVino/openvino_cgvh_dev_2021.2.dockerfile .

2. Run the docker image by sudo docker run -it ubuntu18_dev:2021.2

3. Execute the Object detection sample inside the docker container: ./run_object_detection.sh 
