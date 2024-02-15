# Overview 
## YOLO V8 Object Detection 

This sample application demonstrates how a smart video IoT solution may be created using Intel® hardware and Intel® Distribution of OpenVINO™ Toolkit to perform object detection with a pre-trained YOLO V8 model. This solution detects any number of objects within a video frame looking specifically for known objects.

The results for each frame annotate the input video with boxes around detected objects with a label and a probability value.

For more information about YOLO please see [the documentation here]([https://pjreddie.com/darknet/yolo/](https://github.com/ultralytics/ultralytics))

## How It Works

The YOLO V8 Object Detection application uses the Intel® Distribution of OpenVINO™ toolkit to perform inference on an input video to detect people within each frame. To accomplish this, the application performs the following tasks:

1) Select "Develop AI Application" from HomePage of Developer Toolbox
2) Select YOLOV8 Object Detection by navigating the next arrow button 
3) Select "Benchmark on Intel Hardware" on the far right 
4) Select Hardware by choosing from the drop down list of Filters for "Processors", "Graphics", "Memory" and "Power"
5) Launch the workload
6) Select the "blue downward arrow" under "Application" to view the status of the deployment, it would be one of "Deployed/Running/Completed"
7) Telemetry results and details can be viewed with "Telemetry" button under the "Output" tab
8) Deployment logs and other details  can be viewed with "Deployments" button under the "Output" tab 