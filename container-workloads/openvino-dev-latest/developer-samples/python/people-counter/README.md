# Overview: People Counter System

This sample application illustrates the creation of a smart video IoT solution using Intel® hardware and software tools for precise people counting. The system efficiently detects and tallies the number of individuals within each video frame, providing real-time counts and visually annotating each person with a bounding box.

## How It Works

The sample utilizes an Intel pre-trained model from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), specifically designed as a pedestrian detector tailored for retail scenarios. Built upon a MobileNetV2-like backbone architecture, it incorporates depth-wise convolutions to streamline computation within the 3x3 convolution block. With a single SSD head from a 1/16 scale feature map, it integrates 12 clustered prior boxes for accurate detection.

For more detailed information about this model, refer to the documentation for the [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-retail-0013/README.md) model.

### Deployment Steps

1. Begin by selecting "Develop AI Application" from the HomePage of the Developer Toolbox.
2. Navigate to the next arrow button to opt for YOLOV8 Object Detection.
3. On the far right, select "Benchmark on Intel Hardware".
4. Customize hardware specifications by choosing from the drop-down list for "Processors", "Graphics", "Memory", and "Power".
5. Launch the workload to commence the benchmarking process.
6. Monitor the deployment status via the "blue downward arrow" under "Application", reflecting "Deployed", "Running", or "Completed" states.
7. Access telemetry results and details by clicking the "Telemetry" button under the "Output" tab.
8. For deployment logs and additional insights, click the "Deployments" button under the "Output" tab.

