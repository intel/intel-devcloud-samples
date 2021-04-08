### Intel devcloud sample containerization  with qrpo, application writer libraies

This example introduces the containerized object detection using a pre-trained mobilenet deep learning model  to detect vehicles with qrpo,applicationwriter library calls. This example demonstrates key concepts of OpenVINO 2021.2, to do  the inferencing on Intel® Core™ CPUs.

Following are the steps and commands to build and run the devcloud container.


Steps to run containerized Object detection sample

1. Build the Ubuntu Openvino plus the object detection sample docker image  using the dockerfile/Ubuntu18_OpenVino folder  :
     
	 sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.2/l_openvino_toolkit_dev_ubuntu18_p_2021.2.185.tgz  --build-arg GMMLIB=19.3.2  --build-arg IGC_CORE=1.0.2597  --build-arg IGC_OPENCL=1.0.2597  --build-arg INTEL_OPENCL=19.41.14441  --build-arg INTEL_OCLOC=19.41.14441   --build-arg device="CPU" -t ubuntu18_dev:2021.2 -f ./dockerfiles/ubuntu18/openvino_cgvh_dev_2021.2.dockerfile .
	 
	 alternativel you can execute ./docker.sh 


2.Running the container without Jupyterhub:
      sudo docker run -it ubuntu18_dev:2021.2


3.Running the  container with Jupyterhub: 

  1. Run the docker image by : sudo docker run  -v /tmp:/tmp  -p 8000:8000 --net=host -e DISPLAY  -e DEVICE="CPU" -it ubuntu18_dev:2021.2   /bin/bash
    alternatively you can  execute run_docker.sh 

  2. Execute the Object detection sample inside the docker container on command line :
                                                                    1./run_object_detection.sh
                                                                    2./run_sample.sh 

  3. Execute the Object detection sample inside the docker container using jupyterhub from command line :
       a. Execute jupyterhub (Inside the sample folder)
	   b  Open localhost at port 8000 in the browser with the url as prompted on the jupyterhub console output when running the above[a] command with user/pwd as intel/intel 
	   c. Open the notebook objection_detection_container_ver1.ipynb and execute each cell 

Note:  To run any other container besides object_detection, change the name of the container appropriately