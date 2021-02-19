import torch
from python.utils import get_model
import time
import os
import sys
from qarpo.demoutils import progressUpdate

job_id = os.environ['PBS_JOBID']

def create_onnx_model(model, onnx_filename):
        
    os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)
    
    input_layer_name = "image"
    output_layer_name = ["toolmask"]
    
    dummy_input = torch.randn(1, 3, 1024, 1280)

    with torch.no_grad(): 
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=True, 
                          input_names=[input_layer_name], output_names=output_layer_name)

start_time = time.time()

model_path = "/data/robotic-instrument-segmentation/unet11_binary_20/model_0.pt"
model = get_model(model_path, model_type='UNet11', problem_type='binary')
create_onnx_model(model, "models/onnx/surgical_tools.onnx")

model_path = "/data/robotic-instrument-segmentation/unet11_parts_20/model_0.pt"
model = get_model(model_path, model_type='UNet11', problem_type='parts')
create_onnx_model(model, "models/onnx/surgical_tools_parts.onnx")

# Update progress bar when done
progressUpdate('./results/' + str(job_id) + '.txt', time.time()-start_time, 1, 1)
