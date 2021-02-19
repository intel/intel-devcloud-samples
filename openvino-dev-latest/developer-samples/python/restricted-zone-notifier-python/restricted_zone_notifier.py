"""Restricted Zone Notifier."""

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
import json
import time
import socket
import cv2
import logging as log

from collections import namedtuple
from argparse import ArgumentParser
from inference import Network
from pathlib import Path
from qarpo.demoutils import *
import applicationMetricWriter

# Assemblyinfo contains information about assembly area
MyStruct = namedtuple("assemblyinfo", "safe")
INFO = MyStruct(True)

DELAY = 5

def build_argparser():
    """
    Parse command line arguments.

    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an .xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image. "
                             "'cam' for capturing video stream from camera", )
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                             "CPU, GPU, MYRIAD is acceptable. Looks"
                             " for a suitable plugin for device specified "
                             "(CPU by default)")
    parser.add_argument("-th", "--prob_threshold", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument('-x', '--pointx', default=0, type=int,
                        help="X coordinate of the top left point of assembly"
                             " area on camera feed.")
    parser.add_argument('-y', '--pointy', default=0, type=int,
                        help="Y coordinate of the top left point of assembly"
                             " area on camera feed.")
    parser.add_argument('-w', '--width', default=0, type=int,
                        help="Width of the assembly area in pixels.")
    parser.add_argument('-ht', '--height', default=0, type=int,
                        help="Height of the assembly area in pixels.")
    parser.add_argument('-r', '--rate', default=1, type=int,
                        help="Number of seconds between data updates "
                             "to MQTT server")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)
    parser.add_argument("-nir", "--num_infer_req", help = "Number of Inference Requested", type = int, default = 2)
    return parser


def ssd_out(res, args, initial_wh, selected_region):
    """
    Parse SSD output.

    :param res: Detection results
    :param args: Parsed arguments
    :param initial_wh: Initial width and height of the frame
    :param selected_region: Selected region coordinates
    :return: None
    """
    global INFO
    person = []
    INFO = INFO._replace(safe=True)

    for obj in res:
        # Draw objects only when probability is more than specified threshold
        if obj[2] > args.prob_threshold:
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])
            person.append([xmin, ymin, xmax, ymax])

    for p in person:
        # area_of_person gives area of the detected person
        area_of_person = (p[2] - p[0]) * (p[3] - p[1])
        x_max = max(p[0], selected_region[0])
        x_min = min(p[2], selected_region[0] + selected_region[2])
        y_min = min(p[3], selected_region[1] + selected_region[3])
        y_max = max(p[1], selected_region[1])
        point_x = x_min - x_max
        point_y = y_min - y_max
        # area_of_intersection gives area of intersection of the
        # detected person and the selected area
        area_of_intersection = point_x * point_y
        if point_x < 0 or point_y < 0:
            continue
        else:
            if area_of_person > area_of_intersection:
                # assembly line area flags
                INFO = INFO._replace(safe=True)

            else:
                # assembly line area flags
                INFO = INFO._replace(safe=False)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    global DELAY
    global CLIENT
    global SIG_CAUGHT
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    logger = log.getLogger()
    render_time = 0
    roi_x = args.pointx
    roi_y = args.pointy
    roi_w = args.width
    roi_h = args.height

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        sys.exit(1)

    if input_stream:
        # Adjust DELAY to match the number of FPS of the video file
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.num_infer_req >= video_len:
        num_infer_req = video_len
    else:
        num_infer_req = args.num_infer_req

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, num_infer_req, args.cpu_extension)

    cur_infer_id = 0
    pre_infer_id = 1 - num_infer_req

    frame_count = 0
    job_id = os.environ['PBS_JOBID']
    result_file = open(os.path.join(args.output_dir,f'output_{job_id}.txt'), "w")
    progress_file_path = os.path.join(args.output_dir, f'i_progress_{job_id}.txt')
    infer_time_start = time.time()
    inf_start = 0
    while True:
        dims = ""
        ret, next_frame = cap.read()
        if ret:
            initial_wh = [cap.get(3), cap.get(4)]

            if next_frame is None:
                log.error("ERROR! blank FRAME grabbed")
                break

            # If either default values or negative numbers are given,
            # then we will default to start of the FRAME
            if roi_x <= 0 or roi_y <= 0:
                roi_x = 0
                roi_y = 0
            if roi_w <= 0:
                roi_w = next_frame.shape[1]
            if roi_h <= 0:
                roi_h = next_frame.shape[0]
            key_pressed = cv2.waitKey(int(DELAY))

            selected_region = [roi_x, roi_y, roi_w, roi_h]
            selected_region = [roi_x, roi_y, roi_w, roi_h]
            x_max1= str(selected_region[0])
            x_min1=str(selected_region[0] + selected_region[2])
            y_min1=str(selected_region[1] + selected_region[3])
            y_max1=str(selected_region[1])

            in_frame_fd = cv2.resize(next_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame_fd = in_frame_fd.transpose((2, 0, 1))
            in_frame_fd = in_frame_fd.reshape((n, c, h, w))

            req_time = time.time()
            # Start asynchronous inference for specified request.
            infer_network.exec_net(cur_infer_id, in_frame_fd)
            if pre_infer_id < 0:
                inf_start += (time.time() - req_time)

        if pre_infer_id >= 0:
            inf_wait = time.time()
            # Wait for the result
            infer_network.wait(pre_infer_id)
            # Results of the output layer of the network
            res = infer_network.get_output(pre_infer_id)
            # Parse SSD output
            ssd_out(res, args, initial_wh, selected_region)
            det_time = (time.time() - inf_wait) + inf_start
            applicationMetricWriter.send_inference_time(det_time * 1000)

            est = str(render_time * 1000)
            time1 = round(det_time * 1000)
            Worker = INFO.safe
            out_list = [str(frame_count), x_min1, y_min1, x_max1, y_max1,str(Worker), est, str(time1)]
            for i in range(len(out_list)):
                dims += out_list[i]+' '
            dims += '\n'
            result_file.write(dims)

            render_start = time.time()
            render_end = time.time()
            render_time = render_end - render_start

            frame_count += 1
            if frame_count%10 == 0 or frame_count==video_len-1:
                progressUpdate(progress_file_path, int(time.time()-infer_time_start), frame_count, video_len)

            if frame_count >= video_len:
                break

        cur_infer_id += 1
        pre_infer_id += 1
        
        if cur_infer_id >= num_infer_req:
            cur_infer_id = 0
        if pre_infer_id >= num_infer_req:
            pre_infer_id = 0

        frame = next_frame

        if key_pressed == 27:
            print("Attempting to stop background threads")
            break
    if args.output_dir is None:
        cv2.destroyAllWindows()
    else:
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, f'stats_{job_id}.txt'), 'w') as f:
            f.write('{} \n'.format(round(total_time, 1)))
            f.write('{} \n'.format(frame_count))

    infer_network.clean()
    cap.release()
    cv2.destroyAllWindows()
    applicationMetricWriter.send_application_metrics(args.model, args.device)

if __name__ == '__main__':
    main()
