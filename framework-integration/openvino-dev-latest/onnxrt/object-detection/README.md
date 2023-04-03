# ONNXRuntime - OpenVINO Execution Provider

## How It Works
The sample uses ONNXRuntime OpenVINO EP for the classification. The identified results i.e. classification labels are stored a text file 

* [onnxrt_ovep.dockerfile](dockerfile/ubuntu18/onnxrt_ovep.dockerfile): Utilizes [openvino/onnxruntime_ep_ubuntu18](https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18) as the base image and defines configurable runtime environment variables.
* [run_onnx_objectdetection.sh](run_onnx_objectdetection.sh): Serves as an entrypoint for the container sample. This script run python inference script onnx tiny yolov2 model.
* [ONNX_object_detection.py](ONNX_object_detection.py): Demonstrates inference pipeline on input video file, and saves a log file with inference engine processing time and fps.


## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU_FP32`` | Supports ``GPU_FP32,GPU_FP16,VAD-M_FP16,VAD-F_FP32, or MYRIAD_FP16`` for Openvino EP and ``CPU`` for Default EP |
| ``-v {PATH-TO-HOST-DIR}:/mount_folder`` | PATH-TO-HOST-DIR is the directory to save results. e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}` directory and build:

```
buildah bud --format docker -f ./framework-integration/openvino-dev-latest/onnxrt/object-detection/dockerfile/ubuntu18/onnxrt_ovep.dockerfile -t $REGISTRY_URL/ovep-object-detection:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/ovep-object-detection:custom
```

Navigate to **My Library** > **Resources** and associate the ``ovep-object-detection:custom`` resource with a project, configure the **Output Mount Point** with ``/mount_folder`` and **Environment Variables** with required runtime DEVICE value. Finally click on the launch button.

**NOTE:** 
* For the inference with openvino ep on specific device, configure **Environment Variables** with ``-e DEVICE=[GPU_FP32,GPU_FP16,VAD-M_FP16,VAD-F_FP32, or MYRIAD_FP16]``. For default cpu ep configure with ``-e DEVICE=CPU``.
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU.
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system

```
docker build -t ovep-object-detection:custom -f framework-integration/openvino-dev-latest/onnxrt/object-detection/dockerfile/ubuntu18/onnxrt_ovep.dockerfile .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v {PATH-TO-HOST-DIR}:/mount_folder ovep-object-detection:custom
```
**NOTE:** 
* By defaut the device is CPU_FP32. To enable other device access, use runtime sample config by passing ``-e DEVICE=[device name]``
* You must also mount the device drivers accordingly, see [openvino/onnxruntime_ep_ubuntu18](https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18) for more info.


---
See [README](../../../../README.md) for more info on all marketplace sample applications.
