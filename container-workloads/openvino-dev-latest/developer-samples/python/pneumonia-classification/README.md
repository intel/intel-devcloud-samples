# Healthcare application - Pneumonia Classification Sample

This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and software tools to perform pneumonia classification. This solution uses an inference model that has been trained to classify the presence of pneumonia using a patient's chest X-ray. The results are visualized from what the network has learned using the Class Activation Maps (CAM) technique.

## Background on method

In this application, we use a model trained to classify patients with pneumonia over healthy cases based on their chest X-ray images. The topology used is the DenseNet 121 which is an architecture that has shown to be very efficient at this problem. DenseNet 121 is the first work to claim a classification rate better than practicing radiologists. The dataset used for training is from the "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification" [1] with a CC BY 4.0 license. The trained model is provided as a frozen Tensorflow model.

A Class Activation Map (CAM) [2] is a technique to visualize the regions that are relevant within a Convolutional Neural Network to identify the specific class in the image.

[1] [Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/2) 

[2] Zhou, Bolei, et al. "Learning deep features for discriminative localization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

## How it works
At startup the pneumonia classification application configures itself by parsing the variables from the dockerfile . Once configured, the application loads the specified inference model's IR files into the Inference Engine and runs inference on the specified input X-ray images. The result for each input X-ray image is written to the results.txt file in the form: Pneumonia probability: <*% probability*>, Inference performed in <*time*>, Input file: <*input filename*>

The application uses the following files: 

* [openvino_cgvh_dev_2023.0.0.dockerfile](dockerfile/ubuntu20/openvino_cgvh_dev_2023.0.0.dockerfile): Utilizes [openvino/ubuntu20_dev:2023.0.0](https://hub.docker.com/r/openvino/ubuntu20_dev) as the base image and defines configurable runtime environment variables.
* [classification_pneumonia.py](classification_pneumonia.py): run classification using the DenseNet model.
* [utils.py}(utils.py): Provides image transform utilities 
* [utils_image.py](utils_image.py): Provides code to display the results 
* [run_pneumonia.sh](run_pneumonia.sh): Serves as an entrypoint for the container sample, utilizes the pre-installed model optimizer to convert the mobilenet-ssd [caffe model](resources/worker_safety_mobilenet.caffemodel) before running inference and annotation python scripts.


## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Will support ``FP32`` model precision in upcoming releases. |
| ``-e INPUT_FILE="./validation_images/NORMAL/*.jpeg"`` | Input images inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html), navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud --format docker -f ./developer-samples/python/pneumonia-classification/dockerfile/ubuntu20/openvino_cgvh_dev_2023.0.0.dockerfile -t $REGISTRY_URL/pneumonia-classification:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/pneumonia-classification:custom
```

Navigate to **My Library** > **Resources** and associate the ``pneumonia-classification:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/pneumonia-classification/dockerfile/ubuntu20/openvino_cgvh_dev_2023.0.0.dockerfile -t pneumonia-classification:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -v ./mount_folder:/mount_folder pneumonia-classification:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU``
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu20_dev:2022.3.0](https://hub.docker.com/r/openvino/ubuntu20_dev) for more info.


---
See [README](../../../../../README.md) for more info on all marketplace sample applications.
