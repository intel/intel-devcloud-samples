# People Counter System
This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and software tools to perform people counting. This solution detects and counts the number of people within each video frame displaying the current count of people and drawing a box around each person.

## How It Works
The sample uses an Intel pre-trained model from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).  The model is a pedestrian detector for the Retail scenario. It is based on MobileNetV2-like backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block. The single SSD head from 1/16 scale feature map has 12 clustered prior boxes. For more information about this model see the documentation for [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-retail-0013/README.md) model.

* [openvino_cgvh_dev_2022.1.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2022.1.dockerfile): Utilizes [openvino/ubuntu18_dev](https://hub.docker.com/r/openvino/ubuntu18_dev) as the base image and defines configurable runtime environment variables.
* [run_people_counter.sh](run_people_counter.sh): Serves as an entrypoint for the container sample, uses the model downloader utility to download the person detection model, creates the data output folder, and launches the python inference script.   
* [people_counter.py](people_counter.py): Demonstrates asynchronous inference pipeline on input video file, saves the output video with bounding boxes and labels to output.mp4, and generates a ``perfomance.txt`` file capturing latency and throughput metrics.

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="resources/Pedestrain_Detect_2_1_1.mp4"`` | Input video file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud --format docker -f  ./developer-samples/python/people-counter/dockerfile/ubuntu18/openvino_cgvh_dev_2022.1.dockerfile -t $REGISTRY_URL/people-counter:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/people-counter:custom
```

Navigate to **My Library** > **Resources** and associate the ``people-counter:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/people-counter/dockerfile/ubuntu18/openvino_cgvh_dev_2022.1.dockerfile -t people-counter:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder people-counter:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_dev](https://hub.docker.com/r/openvino/ubuntu18_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.
