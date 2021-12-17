# SAMPLE-NAME-ON-MARKETPLACE
Discription as stated on marketplace. Value-proposition of the Sample.

## How It Works
Describe the start (mount path, inputs) - during -  stop (file outputs) flow of the application.

* [openvino_cgvh_dev_2021.4.dockerfile](dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile): Mention the base image.
* [entripointscript.sh](relative/path/file-name1.py): Describe purpose of file1.
* [file-name2.py](relative/path/file-name2.py): Describe purpose of file2. Include inline links, e.g. utilizes labels listed in [labels.txt](relative/path/labels.txt)

## Runtime Configurations
| Default Config | Description |
| --- | --- |
| ``-e DEVICE=CPU`` | Supports ``GPU`` for running on capable integrated GPU. |
| ``-e PRECISION=FP16`` | Supports ``FP32`` precision |
| ``-e INPUT_FILE={cars_1900.mp4}`` | File path inside the container | 
| ``-e RUN_ON_PREM="/mount_folder"`` | Directory to save results to e.g. mount point to retrieve logs, results |

## Build and run on DevCloud
Using the terminal from the DevCloud [Coding Environment](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index/build-containers-from-terminal.html) terminal, navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
buildah bud -f ./developer-samples/python/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t $REGISTRY_URL/container-name:custom .
```

Push the container to your devcloud private registry:
```
buildah push $REGISTRY_URL/container-name:custom
```

Navigate to **My Library** > **Resources** and associate the ``container-name:custom`` resource with a project, configure the **Mount Point** with ``/mount_folder`` and launch.

**NOTE:** 
* The container playground will ensure GPU access is enabled by default when launching on a device with an integrated GPU. 
* You can indicate the sample to run inference on GPU by passing ``-e DEVICE=GPU`` in **Configuration Parameters**
* Refer to the developer-guide, [configure-imported-containers](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/configure-imported-containers.html)
and [select-hardware-and-launch](https://www.intel.com/content/www/us/en/develop/documentation/devcloud-containers/top/index-2/select-hardware-and-launch.html) for more information.


## Build and run on local system
Navigate to `{repo-root}/container-workloads/openvino-dev-latest` directory and build:
```
docker build -f ./developer-samples/python/object-detection/dockerfile/ubuntu18/openvino_cgvh_dev_2021.4.dockerfile -t container-name:custom .
```

Run the container locally by mounting a local directory to retrieve the results:
```
docker run --rm -it -e RUN_ON_PREM=/mount_folder -v {PATH-TO-HOST-DIRECTORY}:/mount_folder container-name:custom
```
**NOTE:** 
* To enable GPU access, use runtime sample config by passing ``-e DEVICE=GPU`` 
* You must also mount your integrated GPU device e.g.  ``--device /dev/dri:/dev/dri``, see [openvino/ubuntu18_data_dev](https://hub.docker.com/r/openvino/ubuntu18_data_dev) for more info.

---
See [README](../../../../../README.md) for more info on all marketplace sample applications.