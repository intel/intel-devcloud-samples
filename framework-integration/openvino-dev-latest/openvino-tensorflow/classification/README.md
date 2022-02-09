# Opnvino Integration with Tensorflow Classification
Use an optimized and pre-trained inception v3 neural network to classify images. 
 

## How It Works
The sample uses tensorflow APIs and runs inferecne using OpenVINO Inference Engine as backend. The identified results i.e. classification labels are stored a text file 

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Utilizes [openvino/ubuntu18_runtime](https://hub.docker.com/r/openvino/ubuntu18_runtime) as the base image and defines configurable runtime environment variables.
* [convert_yolov4.sh](convert_yolov4.sh): This script is used to generate a tensorflow yolo V4 model
* [run_ovtf_objectdetection.sh](run_ovtf_objectdetection.sh): Serves as an entrypoint for the container sample, utilizes a tensorflow yolo_v4 model[tensorflow model] running inference python scripts with and without Openvino Integration with Tensorflow.
* [object_detection_sample_video_image.py](object_detection_sample_video_image.py): Demonstrates inference pipeline on input image file, and saves an ``output_detections.jpg`` file during execution with resulting bounding box coordinates, detected labels corresponding to IDs from [coco.names](coco.names), detection probabilities along with ``perfomance.txt`` capturing latency and throughput metrics.


## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e INPUT_FILE="./grace_hopper.jpg"`` | Input image file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |
| ``-e FLAG="openvino/native"`` | flag to enable and disable Openvino Integration with Tensorflow optimizations |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:

```
buildah bud --format docker -f ./framework-integration/openvino-dev-latest/openvino-tensorflow/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/ovtf-object-detection:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/ovtf-object-detection:custom
```

Navigate to **My Library** > **Resources** and associate the ``ovtf-object-detection:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./framework-integration/openvino-dev-latest/openvino-tensorflow/classification/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t ovtf-classification:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder object-detection:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.