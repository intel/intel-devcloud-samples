sudo docker build --build-arg package_url=https://storage.openvinotoolkit.org/repositories/openvino/packages/2020.3.2/l_openvino_toolkit_data_dev_ubuntu18_p_2020.3.355.tgz  --build-arg GMMLIB=19.3.2  --build-arg IGC_CORE=1.0.2597  --build-arg IGC_OPENCL=1.0.2597  --build-arg INTEL_OPENCL=19.41.14441  --build-arg INTEL_OCLOC=19.41.14441   --build-arg device="CPU" -t safety_gear:lts -f ./safety-gear-detection-python/dockerfile/ubuntu18/openvino_cgvh_dev_2020.3.2_lts.dockerfile .

 
