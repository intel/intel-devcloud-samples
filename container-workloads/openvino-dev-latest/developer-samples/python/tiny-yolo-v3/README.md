# Tiny YOLO V3 Object Detection

This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and software tools to perform object detection with a pre-trained Tiny YOLO V3 model. This solution detects any number of objects within a video frame looking specifically for known objects.

The results for each frame annotate the input video with boxes around detected objects with a label and a probability value.

For more information about YOLO please see [the documentation here](https://pjreddie.com/darknet/yolo/)

## How It Works

The Tiny YOLO V3 Object Detection application uses the Intel® Distribution of OpenVINO™ toolkit to perform inference on an input video to detect people within each frame. To accomplish this, the application performs the following tasks:

1) Download pre-trained DarkNet Tiny YOLO V3 model weights
2) Convert DarkNet Tiny YOLO V3 model to supported TensorFlow format
3) Convert Tensorflow Tiny YOLO V3 to inference model IR files needed to perform inference
4) Run inference on the provided video file using the OpenVINO inference engine and the converted YOLO IR files. 
5) Writes an output video file with bounding boxes and labels displayed.

The following files are used in the application:

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Utilizes [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) as the base image and defines configurable runtime environment variables.
* [run_tiny_yolo_v3.sh](run_tiny_yolo_v3.sh): Serves as an entrypoint for the container sample. The script:
	* Downloads Tiny YOLO V3 Darknet Model Weights and COCO labels file.
	* Clones the tensorflow-yolo-v3 repository to access the convert_weights_pb.py python script that can convert all different types of YOLO and Tiny YOLO models to frozen Tensorflow Protobuf files
	* Creates the output folder
	* Launches object_detection_demo_yolov3_async.py
* [object_detection_demo_yolov3_async.py](object_detection_demo_yolov3_async.py): Demonstrates asynchronous inference pipeline on input video file, saves the output video with bounding boxes and labels to output.mp4, and generates a ``perfomance.txt`` file capturing latency and throughput metrics.

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/tiny-yolo-v3/classroom.mp4"`` | Input video file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud --format docker -f  ./developer-samples/python/tiny-yolo-v3/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/tiny-yolo-v3:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/tiny-yolo-v3:custom
```

Navigate to **My Library** > **Resources** and associate the ``tiny-yolo-v3:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/tiny-yolo-v3/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/tiny-yolo-v3:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder tiny-yolo-v3:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.