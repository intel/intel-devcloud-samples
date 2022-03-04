# Benchmarking Application 

This sample application demonstrates the OpenVINO(tm) benchmarking utility for a pre-trained model. 
## How It Works

The benchmarking application performs the following steps:

1) Download a pre-trained model using the downloader.py utility. By defualt the model is resnet-50-tf from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) 
2) Convert the model into OpenVINO IR format using the model optimizer utility. 
3) Run benchmarking utility on the model with benchmark_app.py

The following files are used in the application:

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Utilizes [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) as the base image and defines configurable runtime environment variables.
* [benchmark.sh](benchmark.sh): Serves as an entrypoint for the container sample. This script launches all of the operations above. Set the desiered model here. 

* [benchmark_app.sh](benchmark_app.sh): OpenVINO utility that runs the benchmarking on 

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e MODEL="resnet-50-tf"`` | Runs resnet 50 by default|
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="/opt/intel/openvino_$OPENVINO_VERSION/python/samples/tiny-yolo-v3/classroom.mp4"`` | Input video file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud --format docker -f  ./developer-samples/python/benchmark/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/benchmark:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/benchmark:custom
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
docker build -f ./developer-samples/python/benchmark/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/benchmark:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder benchmark:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.
