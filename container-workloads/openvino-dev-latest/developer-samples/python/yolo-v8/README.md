## Overview: YOLO V8 Object Detection 

This sample application demonstrates how a smart video IoT solution can be created using Intel® hardware and the Intel® Distribution of OpenVINO™ Toolkit to perform object detection with a pre-trained YOLO V8 model. This solution efficiently detects any number of objects within a video frame, specifically targeting known objects.

The results for each frame annotate the input video with boxes around detected objects, along with a label and a probability value.

For more information about YOLO, please see [the documentation here](https://pjreddie.com/darknet/yolo/) at the provided link.

## How It Works:

The YOLO V8 Object Detection application utilizes the Intel® Distribution of OpenVINO™ toolkit to perform inference on an input video, detecting objects within each frame. To accomplish this, the application follows these steps:

1) Select "Develop AI Application" from the HomePage of the Developer Toolbox.
2) Navigate to the next arrow button to select YOLOV8 Object Detection.
3) On the far right, select "Benchmark on Intel Hardware".
4) Choose hardware specifications by selecting from the drop-down list of filters for "Processors", "Graphics", "Memory", and "Power".
5) Launch the workload.
6) View the deployment status by selecting the "blue downward arrow" under "Application", which will display as "Deployed", "Running", or "Completed".
7) View telemetry results and details by clicking the "Telemetry" button under the "Output" tab.
8) Access deployment logs and other details by clicking the "Deployments" button under the "Output" tab.
