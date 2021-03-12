#!/usr/bin/env python3

import sys
import io
import cv2
import argparse
import os
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IECore
from qarpo.demoutils import progressUpdate
import applicationMetricWriter

def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", required=False,
                    default='CPU', help="device type")
    ap.add_argument("-i", "--input", required=False,
                    default='.', help="path to input")
    ap.add_argument("-m", "--model", required=False,
                    default='FP32', help="model type")
    ap.add_argument("-o", "--output", required=False,
                    default='noderesult.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    device_type = args['device']
    dir_path = args['input']
    fp_model = args['model']
    output = args['output']

    job_id = os.environ['PBS_JOBID']
    
    # Set up logging
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # Get the input video from the specified path
    cap = cv2.VideoCapture(dir_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up OpenVINO inference
    ie = IECore()
    net = ie.read_network(model='./models/ov/' + fp_model + '/surgical_tools_parts.xml', weights='./models/ov/' + fp_model + '/surgical_tools_parts.bin')
    exec_net = ie.load_network(network=net,device_name=device_type)

    #TODO:replace deprecated layers.keys() with ngraph iterator
    '''if device_type == "CPU":
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = \
            [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by "
                  "the plugin for specified device {}:\n {}".
                  format(ie.device,
                     ', '.join(not_supported_layers)))
            sys.exit(1)'''

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output + f'output_{job_id}.mp4', fourcc, 6.0, (1280, 1024), True)

    infer_time_start = time()
   
    # Run for maximum of 1000 frames
    for number in range(total_frames):

        # Grab the next frame from the video feed, quit if none
        ret, image = cap.read()
        if not ret:
            break

        # Resize the input image to match the expected value
        cropHeight, cropWidth = (1024,1280)
        imgHeight, imgWidth = image.shape[0], image.shape[1] 
        startH = (imgHeight - cropHeight) // 2
        startW = (imgWidth - cropWidth) // 2
        image = image[startH:(startH+cropHeight),startW:(startW+cropWidth),:]

        # Convert from BGR to RGB since model expects RGB input
        rgb_image = image[:,:,np.argsort([2,1,0])]

        # Run the inference
        start_time = time()
        res = exec_net.infer(inputs={"image" : np.expand_dims(np.transpose(rgb_image/255.0, [2, 0, 1]), 0)}) 
        log.info("OpenVINO took {} msec for inference on frame {}".format(1000.0*(time() - start_time), number))
        det_time = time() - start_time
        applicationMetricWriter.send_inference_time(det_time*1000) 
 

        mask_frame = np.zeros((1024,1280,3), dtype=np.uint8)
        frame = res["toolmask"]
       
        sliced = frame[0,1:,:,:]
        mask_frame = (np.floor(np.transpose(sliced, [1,2,0])*255)).astype(np.uint8)

        # Write out frame to video 
        out.write(cv2.addWeighted(image, 1, mask_frame, 0.5, 0))

        progressUpdate(output+f'i_progress_{job_id}.txt', time()-infer_time_start, number+1, total_frames)  

    # Release everything at end of job
    out.release()
    cv2.destroyAllWindows()
    
    # Write time information to the log file
    total_time = time() - infer_time_start
    with open(os.path.join(output, f'stats_{job_id}.txt'), 'w') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(total_frames)+'\n')

    log.info("The output video is {}".format(output + f'output_{job_id}.mp4'))
    model_xml = './models/ov/' + fp_model + '/surgical_tools_parts.xml'
    applicationMetricWriter.send_application_metrics(model_xml, device_type)


if __name__ == '__main__':
    sys.exit(main() or 0)
