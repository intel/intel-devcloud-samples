# Openvino Integration with Torch ORT
Use an optimized and pre-trained Resnet50 neural network to classify images. 
 

## How It Works
The sample uses PyTorch APIs and runs inference using OpenVINO Inference Engine as backend. The identified results i.e. classification labels are stored a text file 

* [openvino_torch_ort.dockerfile](dockerfile/ubuntu18/openvino_torch_ort.dockerfile):  Utilizes [openvino/ubuntu18_runtime](https://hub.docker.com/r/openvino/ubuntu18_runtime) as the base image and defines configurable runtime environment variables.
* [run_ov_torch_ort_image_classification.sh](run_ov_torch_ort_image_classification.sh): Serves as an entrypoint for the container sample, utilizes a pytorch Resnet50 model running inference with and without Openvino Integration with Torch-ORT.
* [resnet_image_classification.py](resnet_image_classification.py): Demonstrates inference pipeline on input image file, and prints detected labels along with ``perfomance.txt`` capturing latency and throughput metrics.


## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e PROVIDER=openvino`` | Runs the samples with OpenVINO Torch-ORT. |
| ``-e BACKEND=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP32`` | Supports FP32 (CPU), FP16 (CPU, GPU). |
| ``-e INPUT_FILE="./data/plane.jpg"`` | Input image file path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:

```
buildah bud --format docker -f ./framework-integration/openvino-dev-latest/openvino-tensorflow/image-classification/dockerfile/ubuntu18/openvino_torch_ort.dockerfile -t $REGISTRY_URL/ov-torch-ort-image-classification:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/ov-torch-ort-image-classification:custom
```

Navigate to **My Library** > **Resources** and associate the ``ov-torch-ort-image-classification:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e BACKEND=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.

## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./framework-integration/openvino-dev-latest/openvino-torch-ort/image-classification/dockerfile/ubuntu18/openvino_torch_ort.dockerfile -t ov-torch-ort-image-classification:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIR}:/mount_folder ov-torch-ort-image-classification:custom
```

---
See [README](../../../../../README.md) for more info on all marketplace sample applications.