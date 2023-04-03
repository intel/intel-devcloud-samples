# YOLO V8 Object Detection

This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and software tools to perform object detection with a pre-trained YOLO V8 model. This solution detects any number of objects within a video frame looking specifically for known objects.

The results for each frame annotate the input video with boxes around detected objects with a label and a probability value.

For more information about YOLO please see [the documentation here]([https://pjreddie.com/darknet/yolo/](https://github.com/ultralytics/ultralytics))

## How It Works

The YOLO V8 Object Detection application uses the Intel® Distribution of OpenVINO™ toolkit to perform inference on an input video to detect people within each frame. To accomplish this, the application performs the following tasks:

1) Run dockerfile to build the application
2) Download the latest yolo-v8 model from ultralytics
3) Run inference on the provided video file using the OpenVINO inference engine and the converted YOLO IR files. 
4) Writes output image files with bounding boxes and labels displayed.

The following files are used in the application:

* [openvino_cgvh_dev_2022.3.dockerfile](dockerfile/ubuntu20/openvino_cgvh_dev_2022.3.dockerfile): Utilizes [openvino/ubuntu20_dev:2022.3.0](https://hub.docker.com/r/openvino/ubuntu20_dev) as the base image and defines configurable runtime environment variables.
* [run_yolo_v8.sh](run_yolo_v8.sh): Serves as an entrypoint for the container sample. The script:
	* Creates the output folder
	* Launches object_detection_demo_yolov8.py
* [object_detection_demo_yolov8.py](object_detection_demo_yolov8.py): Demonstrates inference pipeline on input coco image files, saves the output images with bounding boxes and labels to output folder, and generates a ``perfomance.txt`` file capturing latency and throughput metrics.

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/yolo-v8/images/"`` | Input images file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud --format docker -f  ./developer-samples/python/yolo-v8/dockerfile/ubuntu20/openvino_cgvh_dev_2022.3.dockerfile -t $REGISTRY_URL/yolo-v8:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/yolo-v8:custom
```

Navigate to **My Library** > **Resources** and associate the ``yolo-v8:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/yolo-v8/dockerfile/ubuntu20/openvino_cgvh_dev_2022.3.dockerfile -t $REGISTRY_URL/yolo-v8:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder yolo-v8:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu20_dev:2022.3.0](https://hub.docker.com/r/openvino/ubuntu20_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.
