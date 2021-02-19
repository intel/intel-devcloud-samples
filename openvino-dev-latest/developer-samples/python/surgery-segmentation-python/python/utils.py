import os
import numpy as np
import cv2
import torch
from python.models import UNet16, LinkNet34, UNet11, UNet

def crop_rgb(image, size=(1024,1280)):
    """
    Helper function to crop the image and convert BGR to RGB
    """
    cropHeight, cropWidth = size
    imgHeight, imgWidth = image.shape[0], image.shape[1] 
    startH = (imgHeight - cropHeight) // 2
    startW = (imgWidth - cropWidth) // 2
    return image[startH:(startH+cropHeight),startW:(startW+cropWidth),np.argsort([2,1,0])]

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize single mask on the top of the image
    """
    alpha = 0.5
        
    img = image.copy()
    
    # Flatten if 3D
    if (len(mask.shape) == 3):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ind = mask[:, :] > 0
    
    img[ind,0] = img[ind,0]*alpha + (1-alpha)*color[0] 
    img[ind,1] = img[ind,1]*alpha + (1-alpha)*color[1] 
    img[ind,2] = img[ind,2]*alpha + (1-alpha)*color[2] 
    
    return img

def get_model(model_path, model_type='UNet11', problem_type='binary'):
    """
    :param model_path: Path to the model
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    elif problem_type == 'parts':
        num_classes = 4
    elif problem_type == 'instruments':
        num_classes = 8

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path), map_location=torch.device("cpu"))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    model.eval()

    return model

def create_script(script_path, *lines):
    """
    :param scripts: The name of the script
    :param *lines: The lines to write to the script
    """
    
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

               
    f = open(script_path, "w")

    f.write("cd $PBS_O_WORKDIR\n")

    for line in lines:
        f.write(line)
        f.write("\n")

    f.close()
