# Overview:
## Healthcare application - Pneumonia Classification Sample

This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and software tools to perform pneumonia classification. This solution uses an inference model that has been trained to classify the presence of pneumonia using a patient's chest X-ray. The results are visualized from what the network has learned using the Class Activation Maps (CAM) technique.

## Background on method

In this application, we use a model trained to classify patients with pneumonia over healthy cases based on their chest X-ray images. The topology used is the DenseNet 121 which is an architecture that has shown to be very efficient at this problem. DenseNet 121 is the first work to claim a classification rate better than practicing radiologists. The dataset used for training is from the "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification" [1] with a CC BY 4.0 license. The trained model is provided as a frozen Tensorflow model.

A Class Activation Map (CAM) [2] is a technique to visualize the regions that are relevant within a Convolutional Neural Network to identify the specific class in the image.

[1] [Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/2) 

[2] Zhou, Bolei, et al. "Learning deep features for discriminative localization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

### Deployment Steps

1. Begin by selecting "Develop AI Application" from the HomePage of the Developer Toolbox.
2. Navigate to the next arrow button to opt for YOLOV8 Object Detection.
3. On the far right, select "Benchmark on Intel Hardware".
4. Customize hardware specifications by choosing from the drop-down list for "Processors", "Graphics", "Memory", and "Power".
5. Launch the workload to commence the benchmarking process.
6. Monitor the deployment status via the "blue downward arrow" under "Application", reflecting "Deployed", "Running", or "Completed" states.
7. Access telemetry results and details by clicking the "Telemetry" button under the "Output" tab.
8. For deployment logs and additional insights, click the "Deployments" button under the "Output" tab.
