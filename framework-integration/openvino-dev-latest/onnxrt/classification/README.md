# ONNXRuntime - OpenVINO Execution Provider

## How It Works
The sample uses ONNXRuntime OpenVINO EP for classification. The identified results i.e. classification labels are stored in a text file 

* [onnxrt_ovep.dockerfile](dockerfile/ubuntu18/onnxrt_ovep.dockerfile): Utilizes [openvino/ubuntu18_runtime](https://hub.docker.com/r/openvino/ubuntu18_runtime) as the base image and defines configurable runtime environment variables.
* [run_onnxrt_classification.sh](run_onnxrt_classification.sh): Serves as an entrypoint for the container sample. This script generates the cpp executable and run the same on onnx squeezenet model.
* [squeezenet_cpp_app.cpp](squeezenet_cpp_app.cpp): CPP classification sample scipt.


## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU_FP32`` | Supports ``CPU_FP32, GPU_FP32, GPU_FP16, VAD-M_FP16, VAD-F_FP32 or MYRIAD_FP16`` for OVEP and ``CPU`` for Default EP |
| ``-e EXECUTION_PROVIDER=--use_openvino`` | Uses Openvino EP to execute inference and it supports default CPU when EXECUTION_PROVIDER=--use_cpu  |
| ``-v {PATH-TO-HOST-DIR}:/mount_folder`` | PATH-TO-HOST-DIR is the directory to save results. E.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Use the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html). Download the onnxruntime binaries from the [release page](https://github.com/intel/onnxruntime/releases/tag/v4.0).
```
wget https://github.com/intel/onnxruntime/releases/download/v4.0/linux_binaries_uep_v4.0.tar.gz
```

Unzip the tar file 
```
tar -xzf linux_binaries_uep_v4.0.tar.gz
```

Create a folder "ort-Libraries" at {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification.
```
mkdir {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries
```

Copy libonnxruntime_providers_openvino.so, libonnxruntime_providers_shared.so and libonnxruntime.so.1.11.0 to {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries.
```
cp linux_binaries_uep_v4.0/* {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries
```

Navigate to `{repo-root}` directory and build image:

```
buildah bud --format docker -f ./framework-integration/openvino-dev-latest/onnxrt/classification/dockerfile/ubuntu18/onnxrt_ovep.dockerfile -t $REGISTRY_URL/ovep-classification:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/ovep-classification:custom
```

Navigate to **My Library** > **Resources** and associate the ``ovep-classification:custom`` resource with a project, configure the **Output Mount Point** with ``/mount_folder`` and **Environment Variables** with required runtime DEVICE and EXECUTION_PROVIDER values. Finally click on the launch button.

**NOTE:** 
* For the inference with openvino ep on specific device, configure **Environment Variables** with ``-e DEVICE=[device name]``. For default cpu ep configure with ``-e EXECUTION_PROVIDER=--use_cpu -e DEVICE=CPU``.
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU.
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Download the onnxruntime binaries from the [release page](https://github.com/intel/onnxruntime/releases/tag/v4.0).
```
wget https://github.com/intel/onnxruntime/releases/download/v4.0/linux_binaries_uep_v4.0.tar.gz
```

Unzip the tar file
```
tar -xzf linux_binaries_uep_v4.0.tar.gz
```

Create a folder "ort-Libraries" at {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification.
```
mkdir {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries
```

Copy libonnxruntime_providers_openvino.so, libonnxruntime_providers_shared.so and libonnxruntime.so.1.11.0 to {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries.
```
cp linux_binaries_uep_v4.0/* {repo-root}/framework-integration/openvino-dev-latest/onnxrt/classification/ort-Libraries
```

Navigate to `{repo-root}` directory and build image:

```
docker build -t ovep-classification:custom -f framework-integration/openvino-dev-latest/onnxrt/classification/dockerfile/ubuntu18/onnxrt_ovep.dockerfile  .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v {PATH-TO-HOST-DIR}:/mount_folder ovep-classification:custom
```
**NOTE:** 
* By defaut the device is CPU_FP32. To enable other device access, use runtime sample config by passing ``-e DEVICE=[device name]``
* Pass ``-e EXECUTION_PROVIDER=--use_cpu`` and ``-e DEVICE=CPU`` for executing the inference on CPU EP. By default it is using OpenVINO EP.
* You must also mount the device drivers accordingly, see [openvino/onnxruntime_ep_ubuntu18](https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18) for more info.


---
See [README](../../../../README.md) for more info on all marketplace sample applications.
