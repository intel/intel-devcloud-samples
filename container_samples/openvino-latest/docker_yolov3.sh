sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.3/l_openvino_toolkit_dev_ubuntu18_p_2021.3.394.tgz  --build-arg GMMLIB=19.3.2  --build-arg IGC_CORE=1.0.2597  --build-arg IGC_OPENCL=1.0.2597  --build-arg INTEL_OPENCL=19.41.14441  --build-arg INTEL_OCLOC=19.41.14441   --build-arg device="CPU" -t yolov3:latest -f ./tiny-yolo-v3-python/dockerfile/ubuntu18/openvino_cgvh_dev_2021.3.dockerfile .

 
