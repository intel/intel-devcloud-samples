import sys
import os
from time import time
import cv2
import numpy as np
import torch
from torchvision import transforms, utils
from python.utils import crop_rgb, mask_overlay, get_model
from qarpo.demoutils import progressUpdate

job_id = os.environ['PBS_JOBID']

progress_file='./results/' + str(job_id) + '.txt'
start_time = time()

image = crop_rgb(cv2.imread(str('./data/frame.png')))

mean_values = [0.485, 0.456, 0.406]
scale_values = [0.229, 0.224, 0.225]

progressUpdate(progress_file, time()-start_time, 1, 3)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_values, std=scale_values)])

img_t = transform(image)
batch_t = torch.unsqueeze(img_t, 0)

progressUpdate(progress_file, time()-start_time, 2, 3)

model_path = "/data/robotic-instrument-segmentation/unet11_binary_20/model_0.pt"
model = get_model(model_path, model_type='UNet11', problem_type='binary')

start_time = time()
with torch.no_grad(): 
    res_pytorch = model(batch_t)  # Perform inference with PyTorch Model
    
print("PyTorch took {:,} msec for inference".format(1000.0*(time() - start_time)))

cv2.imwrite("generated/input.png", image)
utils.save_image(res_pytorch, "generated/mask.png")

progressUpdate('./results/' + str(job_id) + '.txt', time()-start_time, 3, 3)
