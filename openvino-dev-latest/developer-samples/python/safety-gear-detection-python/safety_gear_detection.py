#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IECore
from pathlib import Path
from qarpo.demoutils import progressUpdate
import applicationMetricWriter
import ngraph as ng

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Path to an .xml file with a trained model.',
                        required=True,
                        type=str)
    parser.add_argument('-i', '--input',
                        help='Path to video file or image. \'cam\' for capturing video stream from camera.',
                        required=True,
                        type=str)
    parser.add_argument('-ce', '--cpu_extension',
                        help='MKLDNN-targeted custom layers.'
                             'Absolute path to a shared library with the kernel implementation.',
                        type=str,
                        default=None)
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on; CPU, GPU, MYRIAD, or HDDL is acceptable.'
                             'Demo will look for a suitable plugin for specified device (CPU by default).',
                        default='CPU',
                        type=str)
    parser.add_argument('-nireq', '--number_infer_requests',
                        help='Number of parallel inference requests (default is 2).',
                        type=int,
                        required=False,
                        default=2)
    parser.add_argument('-s', '--show',
                        help='Show preview to the user.',
                        action='store_true',
                        required=False)
    parser.add_argument('-l', '--labels',
                        help='Labels mapping file.',
                        default=None,
                        type=str)
    parser.add_argument('-pt', '--prob_threshold',
                        help='Probability threshold for detection filtering.',
                        default=0.5,
                        type=float)
    parser.add_argument('-o', '--output_dir',
                        help='Location to store the results of the processing',
                        default=None,
                        required=True,
                        type=str)
    return parser


def processBoxes(frame_count, res, labels_map, prob_threshold, initial_w, initial_h, result_file):
    for obj in res:
        dims = ""
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            class_id = int(obj[1])
            det_label = labels_map[class_id-1] if labels_map else "class="+str(class_id)
            dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {det_label} {est} {time} \n".format(frame_id=frame_count, xmin=int(obj[3] * initial_w), ymin=int(obj[4] * initial_h), xmax=int(obj[5] * initial_w), ymax=int(obj[6] * initial_h), class_id=class_id, det_label=det_label, est=round(obj[2]*100, 1), time='N/A')
            result_file.write(dims)

class SingleOutputPostprocessor:
    def __init__(self, output_layer):
        self.output_layer = output_layer

    def __call__(self, outputs):
        return outputs[self.output_layer].buffer[0][0]


class MultipleOutputPostprocessor:
    def __init__(self, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.bboxes_layer = bboxes_layer
        self.scores_layer = scores_layer
        self.labels_layer = labels_layer

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer].buffer[0]
        scores = outputs[self.scores_layer].buffer[0]
        labels = outputs[self.labels_layer].buffer[0]
        return [[0, label, score, *bbox] for label, score, bbox in zip(labels, scores, bboxes)]

def get_output_postprocessor(net, bboxes='bboxes', labels='labels', scores='scores'):
    if len(net.outputs) == 1:
        output_blob = next(iter(net.outputs))
        return SingleOutputPostprocessor(output_blob)
    elif len(net.outputs) >= 3:
        def find_layer(name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if name in layer_name]
            if not suitable_layers:
                raise ValueError('Suitable layer for "{}" output is not found'.format(name))

            if len(suitable_layers) > 1:
                raise ValueError('More than 1 layer matched to "{}" output'.format(name))

            return suitable_layers[0]

        labels_out = find_layer(labels, net.outputs)
        scores_out = find_layer(scores, net.outputs)
        bboxes_out = find_layer(bboxes, net.outputs)

        return MultipleOutputPostprocessor(bboxes_out, scores_out, labels_out)

    raise RuntimeError("Unsupported model outputs")

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading plugins for {} device...".format(args.device))
        ie.add_extension(args.cpu_extension, "CPU")

    # Read IR
    log.info("Reading IR...")
    net = ie.read_network(model=model_xml, weights=model_bin)

    #Ensure Model's layer's are supported by MKLDNN
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        ng_function = ng.function_from_cnn(net)
        not_supported_layers = \
                [node.get_friendly_name() for node in ng_function.get_ordered_ops() \
                if node.get_friendly_name() not in supported_layers]

        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert (len(net.input_info.keys()) == 1 or len(net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    for blob_name in net.input_info:
        if len(net.input_info[blob_name].input_data.shape) == 4:
            input_blob = blob_name
        elif len(net.input_info[blob_name].input_data.shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.input_info[blob_name].input_data.shape), blob_name))
    #out_blob = next(iter(net.outputs))
    output_postprocessor = get_output_postprocessor(net)


    # Read and pre-process input image
    n, c, h, w = net.input_info[input_blob].input_data.shape

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=args.number_infer_requests, device_name=args.device)
 

    log.info("Starting preprocessing...")
    job_id = str(os.environ['PBS_JOBID'])
    result_file = open(os.path.join(args.output_dir, f'output_{job_id}.txt'), "w")
    pre_infer_file = os.path.join(args.output_dir, f'pre_progress_{job_id}.txt')
    infer_file = os.path.join(args.output_dir, f'i_progress_{job_id}.txt')
    processed_vid = '/tmp/processed_vid.bin'


    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_len < args.number_infer_requests:
        args.number_infer_requests = video_len 
    #Pre inference processing, read mp4 frame by frame, process using openCV and write to binary file
    width = int(cap.get(3))
    height = int(cap.get(4))
    CHUNKSIZE = n*c*w*h
    id_ = 0
    with open(processed_vid, 'w+b') as f:
        time_start = time.time()
        while cap.isOpened():
            ret, next_frame = cap.read()
            if not ret:
                break
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            bin_frame = bytearray(in_frame) 
            f.write(bin_frame)
            id_ += 1
            if id_%10 == 0: 
                progressUpdate(pre_infer_file, time.time()-time_start, id_, video_len) 
    cap.release()

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    log.info("Starting inference in async mode, {} requests in parallel...".format(args.number_infer_requests))
    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    infer_requests = exec_net.requests
    frame_count = 0

    try:
        infer_time_start = time.time()
        with open(processed_vid, "rb") as data:
            while frame_count < video_len:
                # Read next frame from input stream if available and submit it for inference 
                byte = data.read(CHUNKSIZE)
                if not byte == b"":
                    deserialized_bytes = np.frombuffer(byte, dtype=np.uint8)
                    in_frame = np.reshape(deserialized_bytes, newshape=(n, c, h, w))
                    inf_time = time.time()
                    exec_net.start_async(request_id=current_inference, inputs={input_blob: in_frame})
                    det_time = time.time() - inf_time
                    applicationMetricWriter.send_inference_time(det_time*1000)         
                
                # Retrieve the output of an earlier inference request
                if previous_inference >= 0:
                    status = infer_requests[previous_inference].wait()
                    if status is not 0:
                        raise Exception("Infer request not completed successfully")
                    # Parse inference results
                    det_time = time.time() - inf_time
                    applicationMetricWriter.send_inference_time(det_time*1000)                      
                    res = output_postprocessor(exec_net.requests[previous_inference].output_blobs)
                    processBoxes(frame_count, res, labels_map, args.prob_threshold, width, height, result_file)
                    frame_count += 1

                # Write data to progress tracker
                if frame_count % 10 == 0: 
                    progressUpdate(infer_file, time.time()-infer_time_start, frame_count+1, video_len+1) 

                # Increment counter for the inference queue and roll them over if necessary 
                current_inference += 1
                if current_inference >= args.number_infer_requests:
                    current_inference = 0

                previous_inference += 1
                if previous_inference >= args.number_infer_requests:
                    previous_inference = 0

        # End while loop
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, f'stats_{job_id}.txt'), 'w') as f:
            f.write('{:.3g} \n'.format(total_time))
            f.write('{} \n'.format(frame_count))

        result_file.close()
    
    
    finally:
        log.info("Processing done...")
        applicationMetricWriter.send_application_metrics(model_xml, args.device)
        del exec_net
        
    applicationMetricWriter.send_application_metrics(model_xml, args.device)

if __name__ == '__main__':
    sys.exit(main() or 0)
