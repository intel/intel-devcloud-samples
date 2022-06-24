"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log

from argparse import ArgumentParser
from inference import Network
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
#from qarpo.demoutils import *

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)
    parser.add_argument("-nir", "--num_infer_req", help = "Number of inference requests", type = int, default = None)
    return parser


def performance_counts(perf_count):
    """
    print information about layers of the model.

    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


def ssd_out(frame, result):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count


def main():
    """
    Load the network and parse the SSD output.

    :return: None
    """

    args = build_argparser().parse_args()

    # Flag for the input image
    single_image_mode = False
    total_count = 0
    last_count = 0
    total_count = 0
    start_time = 0
    load_time = 0
    frame_count = 0
    result_list = []

    # Checks for input image
    if args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_infer_req = args.num_infer_req

    if video_len < num_infer_req:
        num_infer_req = video_len

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          num_infer_req, args.cpu_extension)

    cur_request_id = 0
    pre_request_id = 1 - num_infer_req

    #job_id = os.environ['PBS_JOBID']
    progress_file_path = os.path.join(args.output_dir, f'i_progress.txt')
    post_progress_file_path = os.path.join(args.output_dir, f'post_progress.txt')

    infer_time_start = time.time()
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    people_counter = cv2.VideoWriter(os.path.join(args.output_dir, f"output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(initial_w), int(initial_h)), True)
    while True:
        if frame_count < video_len:
            flag, frame = cap.read()
            frame_count += 1
            if not flag:
                break
            # Start async inference
            image = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            image = image.reshape((n, c, h, w))
            # Start asynchronous inference for specified request.
            inf_start = time.time()
            infer_network.exec_net(cur_request_id, image)
            if frame_count <= num_infer_req:
                load_time += (time.time() - inf_start)

        if pre_request_id >= 0:
            inf_start = time.time()
            # Wait for the result
            if infer_network.wait(pre_request_id) == 0:
                # Results of the output layer of the network
                det_time = (time.time() - inf_start) + (load_time / num_infer_req)
                result = infer_network.get_output(pre_request_id)
                result_list.append([result, det_time*1000])

                if args.perf_counts:
                    perf_count = infer_network.performance_counter(pre_request_id)
                    performance_counts(perf_count)

                #if len(result_list)%10 == 0 or len(result_list) == video_len:
                    #progressUpdate(progress_file_path, time.time()-infer_time_start, frame_count, video_len)

                if len(result_list) == video_len:
                    break

        pre_request_id += 1
        if pre_request_id >= num_infer_req:
            pre_request_id = 0

        cur_request_id += 1
        if cur_request_id >= num_infer_req:
            cur_request_id = 0

    if args.output_dir:
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, f'performance.txt'), 'w') as f:
            f.write('Throughput: {:.3g} FPS \n'.format(frame_count/total_time))
            f.write('Latency: {:.3f} ms\n'.format(total_time*1000))

    cap.release()
    infer_network.clean()
    del (frame_count)
    del (infer_time_start)

    # Post Progress
    post_processing_time_start = time.time()
    frame_count = 0
    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print ("Video Len:",video_len)
    while True:
        ret, frame = cap.read()
        if ret == False:
            print ("Break")
            cap.release()
            break

        frame, current_count = ssd_out(frame, result_list[frame_count][0])
        inf_time_message = "Inference time: {:.3f}ms"\
                           .format(result_list[frame_count][1])
        cv2.putText(frame, inf_time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        current_count_message = "Current count: {}"\
                                 .format(current_count)
        cv2.putText(frame, current_count_message, (15, 30),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        last_count = current_count
        people_counter.write(frame)
        frame_count += 1
        #if frame_count%10 == 0:
            #progressUpdate(post_progress_file_path, time.time()-post_processing_time_start, frame_count, video_len)


if __name__ == '__main__':
    main()
    exit(0)
