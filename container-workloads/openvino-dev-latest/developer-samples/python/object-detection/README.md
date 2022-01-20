# Object Detection
Use an optimized and pre-trained MobileNet-SSD neural network to detect vehicles in a pre-recorded video clip. 

## How It Works
The sample converts a **mobilenet-ssd** caffe model for optimized inference and feeds a video frame-by-frame to the OpenVINO Inference Engine. The identified results i.e. detected vehicles are stored to a text file which are used later to annotate all the frames of the original video.

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Utilizes [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) as the base image and defines configurable runtime environment variables.
* [run_object_detection.sh](run_object_detection.sh): Serves as an entrypoint for the container sample, utilizes the pre-installed model optimizer to convert the mobilenet-ssd [caffe model](mobilenet-ssd/mobilenet-ssd.caffemodel) before running inference and annotation python scripts.
* [object_detection.py](object_detection.py): Demonstrates asynchronous inference pipeline on input video file, and saves an ``output.txt`` file during execution with resulting bounding box coordinates, detected labels corresponding to IDs from [labels.txt](labels.txt), detection probabilities along with ``perfomance.txt`` capturing latency and throughput metrics.
* [object_detection_annotate.py](safety_gear_detection_annotate.py): Reads the original video file, annotates frame-by-frame inference results (e.g. bounding boxes, label text) and saves a new output.mp4 file after execution.

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="resources/Safety_Full_Hat_and_Vest.mp4"`` | Input video file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:

```
buildah bud --format docker -f ./developer-samples/python/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/object-detection:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/object-detection:custom
```

Navigate to **My Library** > **Resources** and associate the ``object-detection:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t object-detection:custom .
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