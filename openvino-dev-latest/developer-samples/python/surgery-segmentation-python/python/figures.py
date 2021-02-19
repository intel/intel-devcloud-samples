import os
import time
import sys
import cv2
import numpy as np
from pathlib import Path
import logging as log
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore
from qarpo.demoutils import progressUpdate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from python.utils import crop_rgb, mask_overlay

# Set up logging
log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

# Input Image
image = cv2.imread(str('./data/frame.png'))

# Left Tool Segmentation
mask_left = cv2.imread(str('./data/left_frame.png'))

# Right Tool Segmentation
mask_right = cv2.imread(str('./data/right_frame.png'))


# Convert image and mask to 1024x1280x3 RGB
image = crop_rgb(image)
mask = crop_rgb(mask_left + mask_right)

# Load surgical_tools network
ie_surgical_tools = IECore()
net_surgical_tools = ie_surgical_tools.read_network(model="models/ov/FP16/surgical_tools.xml", weights="models/ov/FP16/surgical_tools.bin")
exec_net_surgical_tools = ie_surgical_tools.load_network(network=net_surgical_tools, device_name='CPU')

# Infer
start_time = time.time()
res_ov_surgical_tools = exec_net_surgical_tools.infer(inputs={"image" : np.expand_dims(np.transpose(image/255.0, [2, 0, 1]), 0)})
log.info("OpenVINO took {} msec for inference".format(1000.0*(time.time() - start_time)))

# Load surgical_tools_parts network
ie_surgical_tools_parts = IECore()
net_surgical_tools_parts = ie_surgical_tools_parts.read_network(model="models/ov/FP16/surgical_tools_parts.xml", weights="models/ov/FP16/surgical_tools_parts.bin")
exec_net_surgical_tools_parts = ie_surgical_tools_parts.load_network(network=net_surgical_tools_parts, device_name='CPU')

# Infer
start_time = time.time()
res_ov_surgical_tools_parts = exec_net_surgical_tools_parts.infer(inputs={"image" : np.expand_dims(np.transpose(image/255.0, [2, 0, 1]), 0)})
log.info("OpenVINO took {} msec for inference".format(1000.0*(time.time() - start_time)))

# Plot Results
fig = plt.figure(figsize=(20,20));

# Binary segmentation figures
output_ov_surgical_tools = res_ov_surgical_tools["toolmask"][0,0]
predicted = np.zeros((1024,1280,3), dtype=np.uint8)
predicted[output_ov_surgical_tools > 0, 0] = 255
mask[mask[:,:,0] > 0] = [0,0,255]

plt.subplot(3,3,1)
plt.imshow(output_ov_surgical_tools);
plt.title("OpenVINO Prediction");
ax = plt.gca();
ax.set_ylabel("Surgical Tools Network", rotation=90, size='large')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)

plt.subplot(3,3,2)
plt.imshow(predicted+mask);
plt.title("Ground Truth (Blue), Predicted (Red), Overlap (Magenta)");

overlay = mask_overlay(image, output_ov_surgical_tools)
plt.subplot(3,3,3)
plt.imshow(overlay);
plt.title("Example segmentation")


# Parts segmentation figures
output_ov_surgical_tools_parts = res_ov_surgical_tools_parts["toolmask"]

img = image.copy()

start_time = time.time()

sliced = output_ov_surgical_tools_parts[0,[3,2,1],:,:]
predicted_ov_surgical_tools_parts = (np.floor(np.transpose(sliced, [1,2,0])*255)).astype(np.uint8)

plt.subplot(3,3,4)
plt.imshow(np.absolute(output_ov_surgical_tools_parts[0,0]-1));
plt.title("OpenVINO prediction");
ax = plt.gca();
ax.set_ylabel("Surgical Tools Parts Network", rotation=90, size='large')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)

plt.subplot(3,3,5)
plt.imshow(predicted_ov_surgical_tools_parts);
plt.title("Part 1 (Red), Part 2 (Green), Part 3 (Blue)");

plt.subplot(3,3,6)
predicted 
ind0 = predicted_ov_surgical_tools_parts[:, :, 0] > 0 
ind1 = predicted_ov_surgical_tools_parts[:, :, 1] > 0
ind2 = predicted_ov_surgical_tools_parts[:, :, 2] > 0 
img[ind0] = img[ind0]*.5 + .5*predicted_ov_surgical_tools_parts[ind0]
img[ind1] = img[ind1]*.5 + .5*predicted_ov_surgical_tools_parts[ind1]
img[ind2] = img[ind2]*.5 + .5*predicted_ov_surgical_tools_parts[ind2]
plt.imshow(img);
plt.title("Example segmentation")

log.info("OpenVINO took {} msec for image processing alternate".format(1000.0*(time.time() - start_time)))

fig.savefig('generated/predictions.png', dpi=fig.dpi, bbox_inches='tight')

# Update progress bar when done
job_id = os.environ['PBS_JOBID']

progressUpdate('./results/' + str(job_id) + '.txt', time.time()-start_time, 1, 1)
