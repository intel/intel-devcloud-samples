# Deep Learning Streamer
Construct a media analytics pipeline using OpenVINO™ and GStreamer to create optimized inference and media operations. 

## How It Works

[GStreamer](https://gstreamer.freedesktop.org/) is a flexible, fast and multiplatform open-source multimedia framework. It has an easy to use command line tool for running  pipelines, as well as an API with bindings in C, Python, Javascript and [more](https://gstreamer.freedesktop.org/bindings/).
This sample uses the GStreamer command line tool `gst-launch-1.0`. For more information and examples please refer to the online documentation [gst-launch-1.0](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c).  

### Pipelines
The command line tool `gst-launch-1.0` enables developers to describe media analytics pipeline as a series of connected elements. The list of elements, their configuration properties, and their connections are all specified as a list of strings seperated by exclamation marks (!). `gst-launch-1.0` parses the string and instantiates the software modules which perform the individual media analytics operations. Internally the GStreamer library constructs a pipeline object that contains the individual elements and handles common operations such as clocking, messaging, and state changes.

**Example**:
```gst-launch-1.0 videotestsrc ! ximagesink```

### Elements
An [element](https://gstreamer.freedesktop.org/documentation/application-development/basics/elements.html?gi-language=c) is the fundamental building block of a pipeline. Elements perform specific operations on incoming frames and then push the resulting frames downstream for further processing. Elements are linked together textually by exclamation marks (`!`) with the full chain of elements representing the entire pipeline. Each element will take data from its upstream element, process it and then output the data for processing by the next element.

Elements designated as **source** elements provide input into the pipeline from external sources. In this tutorial we use the [filesrc](https://gstreamer.freedesktop.org/documentation/coreelements/filesrc.html?gi-language=c#filesrc) element that reads input from a local file.  

Elements designated as **sink** elements represent the final stage of a pipeline. As an example, a sink element could write transcoded frames to a file on the local disk or open a window to render the video content to the screen or even restream the content via rtsp. In the benchmarking section of this tutorial our primary focus will be to compare the performance of media analytics pipelines on different types of hardware and so we will use the standard [fakesink](https://gstreamer.freedesktop.org/documentation/coreelements/fakesink.html?gi-language=c#fakesink) element to end the pipeline immediately after the analytics is complete without further processing.

We will also use the [decodebin](https://gstreamer.freedesktop.org/documentation/playback/decodebin.html#decodebin) utility element. The `decodebin` element constructs a concrete set of decode operations based on the given input format and the decoder and demuxer elements available in the system. At a high level the decodebin abstracts the individual operations required to take encoded frames and produce raw video frames suitable for image transformation and inferencing.

The next step in the pipeline after decoding is color space conversion which is handled by the [videoconvert](https://gstreamer.freedesktop.org/documentation/videoconvert/index.html?gi-language=c#videoconvert) element. The exact transformation required is specified by placing a [capsfilter](https://gstreamer.freedesktop.org/documentation/coreelements/capsfilter.html?gi-language=c#capsfilter) on the output of the videoconvert element. In this case we specify BGRx because this is the format used by the detection model.
<a id='dl-streamer'></a>
#### DL Streamer elements
Elements that start with the prefix 'gva' are from DL Streamer and are provided as part of the OpenVINO™ toolkit. There are five DL Streamer elements used in this tutorial which we will describe here along with the properties that will be used. Refer to [DL Streamer elements page](https://github.com/opencv/gst-video-analytics/wiki/Elements) for the list of all DL Streamer elements and usages.  

* [gvadetect](https://github.com/opencv/gst-video-analytics/wiki/gvadetect) - Runs detection with the OpenVINO™ inference engine. We will use it to detect vehicles in a frame and output their bounding boxes.
	- `model` - path to the inference model network file.
	- `device` - device to run inferencing on. 
	- `inference-interval` - interval between inference requests, the bigger the value, the better the throughput. i.e. setting this property to 1 would mean run deteciton on every frame while setting it to 5 would run detection on every fifth frame.
* [gvaclassify](https://github.com/opencv/gst-video-analytics/wiki/gvaclassify) - Runs classification with the OpenVINO™ inference engine. We will use it to label the bounding boxes that `gvadetect` output with the type and color of the vehicle. 
	- `model` - path to the inference model network file.
	- `model-proc` - path to the model-proc file. More information on what a model-proc file is can be found in [section 2.4](#model-proc).
	- `device` - device to run inferencing on. 
    - `reclassify-interval` - How often to reclassify tracked objects. Only valid when used with `gvatrack`.
* [gvawatermark](https://github.com/opencv/gst-video-analytics/wiki/gvawatermark) - Overlays detection and classification results on top of video data. We will do exeactly that. Parse the detected vehicle results metadata and create a video frame rendered with the bounding box aligned to the vehicle position; parse the classified vehicle result and label it on the bounding box.  
* [gvafpscounter](https://github.com/opencv/gst-video-analytics/wiki/gvafpscounter) - Measure Frames Per Second across multiple streams and print to the output. 
	- `starting-frame` specifies the frame to start collecting fps measurements. In this tutorial, we start at frame 10 to not include initialization time in our performance output.
* [gvatrack](https://github.com/opencv/gst-video-analytics/wiki/gvatrack) - Identifies objects in frames where detection is skipped. This allows us to run object detection on fewer frames and increases overall throughput while still tracking the position and type of objects in every frame.

### Properties
Elements are configured using key, value pairs called properties. As an example the filesrc element has a property named `location` which specifies the file path for input.

**Example**:
 ```filesrc location=cars_1900.mp4```.

The documentation for each element (which can be viewed using the command line tool `gst-inspect-1.0`) describes its properties as well as the valid range of values for each property.

## Important Files

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Utilizes [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) as the base image and defines configurable runtime environment variables.
* [run_dlstreamer_tutorial.sh](run_dlstreamer_tutorial.sh): Serves as an entrypoint for the container sample, downloads pedestrian-and-vehicle-detector and vehicle-attributes-recognition FP16 models, runs:
	* vehicle_detection_and_classification.sh
	* vehicle_detection_benchmark.sh
	* vehicle_tracking_benchmark.sh
	
* [vehicle_detection_and_classification.sh](vehicle_detection_and_classification.sh): Runs object detection pipeline with gst-launch-1.0
* [vehicle_detection_benchmark.sh](vehicle_detection_benchmark.sh): Runs vehicle detection benchmarking pipeline with gst-launch-1.0
* [vehicle_tracking_benchmark.sh] (vehicle_tracking_benchmark.sh): Runs vehicle tracking pipeline with gst-launch-1.0

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
buildah bud --format docker -f ./tutorials/python/dlstreamer/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/dl-streamer:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/dl-streamer:custom
```

Navigate to **My Library** > **Resources** and associate the ``dl-streamer:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./tutorials/python/dlstreamer/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/dl-streamer:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder dl-streamer:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.