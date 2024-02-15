
## Overview: Benchmarking Application 

This sample application illustrates the usage of OpenVINO™'s benchmarking utility for evaluating the performance of a pre-trained model.

## How It Works:

The benchmarking application executes the following steps:

1. **Model Download**: Utilizes the `downloader.py` utility to download a pre-trained model. By default, the model selected is `resnet-50-tf` from the Open Model Zoo.
   
2. **Model Conversion**: Converts the downloaded model into the OpenVINO Intermediate Representation (IR) format using the Model Optimizer utility.
   
3. **Benchmarking Execution**: Runs the benchmarking utility (`benchmark_app.py`) on the converted model.
   
   1) Begin by selecting "Develop AI Application" from the Developer Toolbox homepage.
   
   2) Proceed by navigating to the next arrow button to opt for the YOLOV8 Object Detection.
   
   3) On the far right, opt for "Benchmark on Intel Hardware".
   
   4) Customize hardware specifications by choosing from the drop-down list for "Processors", "Graphics", "Memory", and "Power".
   
   5) Launch the workload to commence the benchmarking process.
   
   6) Monitor the deployment status via the "blue downward arrow" under "Application", reflecting "Deployed", "Running", or "Completed" states.
   
   7) Access telemetry results and details through the "Telemetry" button under the "Output" tab.
   
   8) For deployment logs and additional insights, click the "Deployments" button under the "Output" tab.




