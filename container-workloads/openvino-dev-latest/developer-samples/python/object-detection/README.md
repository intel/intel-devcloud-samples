# Overview: Object Detection
Use an optimized and pre-trained MobileNet-SSD neural network to detect vehicles in a pre-recorded video clip. 

## How It Works
The sample converts a **mobilenet-ssd** caffe model for optimized inference and feeds a video frame-by-frame to the OpenVINO Inference Engine. The identified results i.e. detected vehicles are stored to a text file which are used later to annotate all the frames of the original video.

### Deployment Steps

1. Begin by selecting "Develop AI Application" from the HomePage of the Developer Toolbox.
2. Navigate to the next arrow button to opt for YOLOV8 Object Detection.
3. On the far right, select "Benchmark on Intel Hardware".
4. Customize hardware specifications by choosing from the drop-down list for "Processors", "Graphics", "Memory", and "Power".
5. Launch the workload to commence the benchmarking process.
6. Monitor the deployment status via the "blue downward arrow" under "Application", reflecting "Deployed", "Running", or "Completed" states.
7. Access telemetry results and details by clicking the "Telemetry" button under the "Output" tab.
8. For deployment logs and additional insights, click the "Deployments" button under the "Output" tab.
