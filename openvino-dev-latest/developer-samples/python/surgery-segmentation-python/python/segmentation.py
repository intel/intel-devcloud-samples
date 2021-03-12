from __future__ import print_function
import sys
import io
import cv2
import argparse
import os
import numpy as np
import logging as log
import ngraph as ng
from time import time
from openvino.inference_engine import IECore
from qarpo.demoutils import progressUpdate

def crop(image, size=(1024,1280)):
    """
    Helper function to crop the image
    """
    cropHeight, cropWidth = size
    imgHeight, imgWidth = image.shape[0], image.shape[1] 
    startH = (imgHeight - cropHeight) // 2
    startW = (imgWidth - cropWidth) // 2
    return image[startH:(startH+cropHeight),startW:(startW+cropWidth),:]

def main():
    # Set up logging
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", required=False,
                    default='CPU', help="device type")
    ap.add_argument("-i", "--input", required=False,
                    default='input.mp4', help="path to input")
    ap.add_argument("-m", "--model", required=False,
                    default='FP32', help="model type")
    ap.add_argument("-o", "--output", required=False,
                    default='results/', help="output directory")
    args = vars(ap.parse_args())
    
    # Arguments
    device_type = args['device']
    path = args['input']
    fp_model = args['model']
    output = args['output']
    
    job_id = os.environ['PBS_JOBID']
    
    # Get the input video from the specified path
    log.info("Fetching video from {}".format(path))
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up OpenVINO inference
    ie = IECore()
    net = ie.read_network(model='./ov_models/' + fp_model + '/surgical_tools.xml', weights='./ov_models/' + fp_model + '/surgical_tools.bin')
    exec_net = ie.load_network(network=net,device_name=device_type)

    # Ensure layers are supported by CPU plugin
    if device_type == "CPU":
        supported_layers = ie.query_network(net,device_type)
        ng_function = ng.function_from_cnn(net)
        not_supported_layers = \
                [node.get_friendly_name() for node in ng_function.get_ordered_ops() \
                if node.get_friendly_name() not in supported_layers]
                
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by "
                  "the plugin for specified device {}:\n {}".
                  format(ie.device,
                     ', '.join(not_supported_layers)))
            sys.exit(1)

    # Create the VideoWriter object
    out = cv2.VideoWriter(output + 'output_' + str(job_id) + '.mp4', cv2.VideoWriter_fourcc(*"avc1"), 6.0, (1280, 1024), True)
            
    infer_time_start = time()
    
    # Process all of the frames
    for number in range(total_frames):

        # Grab the next frame from the video feed, quit if none
        ret, image = cap.read()
        if not ret:
            break

        # Resize the input image to match the expected value
        image = crop(image)
        image_rgb = image[:,:,np.argsort([2,1,0])]

        # Run the inference
        start_time = time()
        res = exec_net.infer(inputs={"image" : np.expand_dims(np.transpose(image_rgb/255.0, [2, 0, 1]), 0)}) 
        log.info("OpenVINO took {} msec for inference on frame {}".format(1000.0*(time() - start_time), number))

        # Create a mask using the predicted classes	
        mask_frame = np.zeros((1024,1280,3), dtype=np.uint8)
        frame = res["toolmask"][0,0]
        mask_frame[frame > 0] = [0,255,0] 
               
        # Write out frame to video 
        out.write(cv2.addWeighted(image, 1, mask_frame, 0.5, 0))

        # Update the progress file
        progressUpdate('./results/' + str(job_id) + '.txt', time()-infer_time_start, number+1, total_frames)       

    # Release everything at end of job
    out.release()
    cv2.destroyAllWindows()
    
    # Write time information to the log file
    total_time = time() - infer_time_start
    with open(os.path.join(output, 'stats_'+str(job_id)+'.txt'), 'w') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(total_frames)+'\n')

    log.info("The output video is {}".format(output + 'output_' + str(job_id) + '.mp4'))

if __name__ == '__main__':
    sys.exit(main() or 0)
