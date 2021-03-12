"""
 Authors: Stefan Andritoiu <stefan.andritoiu@gmail.com>
          Serban Waltter <serban.waltter@rinftech.com>
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

##########################################################
# INCLUDES
##########################################################

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy
import time
import datetime
import collections
import threading
import datetime
import math
from openvino.inference_engine import IECore
from pathlib import Path
from qarpo.demoutils import progressUpdate
import applicationMetricWriter



##########################################################
# CONSTANTS
##########################################################

#CONF_FILE = './resources/conf.txt'
CONF_VIDEODIR = './UI/resources/video_frames/'
CONF_DATAJSON_FILE = './UI/resources/video_data/data.json'
CONF_VIDEOJSON_FILE = './UI/resources/video_data/videolist.json'
CPU_EXTENSION = ''
#TARGET_DEVICE = 'CPU'
STATS_WINDOW_NAME = 'Statistics'
CAM_WINDOW_NAME_TEMPLATE = 'inference_output_Video_{}_{}'
PROB_THRESHOLD = 0.145
FRAME_THRESHOLD = 5
WINDOW_COLUMNS = 3
LOOP_VIDEO = False
UI_OUTPUT = False

##########################################################
# GLOBALS
##########################################################

model_xml = ''
model_bin = ''
labels_file = ''
videoCaps = []
display_lock = threading.Lock()
log_lock = threading.Lock()
frames = 0
frameNames = []
numVids = 20000

##########################################################
# CLASSES
##########################################################
class FrameInfo:
    def __init__(self, frameNo=None, count=None, timestamp=None):
        self.frameNo = frameNo
        self.count = count
        self.timestamp = timestamp

class VideoCap:
    def __init__(self, cap, req_label, cap_name, is_cam):
        self.cap = cap
        self.req_label = req_label
        self.cap_name = cap_name
        self.is_cam = False
        self.cur_frame = {}
        self.initial_w = 0
        self.initial_h = 0
        self.frames = 0
        self.cur_frame_count = 0
        self.total_count = 0
        self.last_correct_count = 0
        self.candidate_count = 0
        self.candidate_confidence = 0
        self.closed = False
        self.countAtFrame = []
        self.video = None
        self.rate = 0
        self.start_time = {}

        if not is_cam:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            self.fps = 0

        self.videoName = cap_name + ".mp4"

    def init_vw(self, h, w, fps):
        self.video = cv2.VideoWriter(os.path.join(output_dir, self.videoName), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h), True) 
        if not self.video.isOpened():
            print ("Could not open for write" + self.videoName)
            sys.exit(1)


##########################################################
# FUNCTIONS
##########################################################

def env_parser():
    global TARGET_DEVICE, numVids, LOOP_VIDEO
    if 'DEVICE' in os.environ:
        TARGET_DEVICE = os.environ['DEVICE']

    if 'LOOP' in os.environ:
        lp = os.environ['LOOP']
        if lp == "true":
            LOOP_VIDEO = True
        if lp == "false":
            LOOP_VIDEO = False

    if 'NUM_VIDEOS' in os.environ:
        numVids = int(os.environ['NUM_VIDEOS'])

def args_parser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU or MYRIAD is acceptable. Application "
                             "will look for a suitable plugin for device specified (CPU by default)", type=str)
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model's weights.", required=True, type=str)
    parser.add_argument("-l", "--labels", help="Labels mapping file", default=None, type=str, required=True)
    parser.add_argument("-e", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-lp", "--loop", help = "Loops video to mimic continous input", type = str, default = None)
    parser.add_argument("-c", "--config_file", help = "Path to config file", type = str, default = None)
    parser.add_argument("-n", "--num_videos", help = "Number of videos to process", type = int, default = None)
    parser.add_argument("-nr", "--num_requests", help = "Number of inference requests running in parallel", type = int, default = None)
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)

    global model_xml, model_bin, device, labels_file, CPU_EXTENSION, LOOP_VIDEO, config_file, num_videos, output_dir, num_infer_requests

    args = parser.parse_args()
    if args.model:
        model_xml = args.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
    if args.labels:
        labels_file = args.labels
    if args.device:
        device = args.device
    if args.cpu_extension:
        CPU_EXTENSION = args.cpu_extension
    if args.loop:
        lp = args.loop
        if lp == "true":
            LOOP_VIDEO = True
        if lp == "false":
            LOOP_VIDEO = False
    if args.config_file:
        config_file = args.config_file
    if args.num_videos:
        num_videos = args.num_videos
    if args.num_requests:
        num_infer_requests = args.num_requests
    if args.output_dir:
        output_dir = args.output_dir


def check_args(defaultTarget=None):
    # ArgumentParser checks model and labels by default right now
    if model_xml == '':
        print ("You need to specify the path to the .xml file")
        print ("Use -m MODEL or --model MODEL")
        sys.exit(11)
    if labels_file == '':
        print ("You need to specify the path to the labels file")
        print ("Use -l LABELS or --labels LABELS")
        sys.exit(12)
    with open(config_file, 'r') as f:
        if not f.read():
            print("Please use at least 1 video")
            sys.exit(13)

def parse_conf_file(job_id):
    """
        Parses the configuration file.
        Reads videoCaps
    """
    with open(config_file, 'r') as f:
        cnt = 0
        for idx, item in enumerate(f.read().splitlines()):
            if cnt < num_videos:
                split = item.split()
                if split[0].isdigit():
                    videoCap = VideoCap(cv2.VideoCapture(int(split[0])), split[1], CAM_WINDOW_NAME_TEMPLATE.format(job_id, idx), True)
                else:
                    if os.path.isfile(split[0]) :
                        videoCap = VideoCap(cv2.VideoCapture(split[0]), split[1], CAM_WINDOW_NAME_TEMPLATE.format(job_id, idx), False)
                    else:
                        print ("Couldn't find " + split[0])
                        sys.exit(3)
                videoCaps.append(videoCap)
                cnt += 1
            else:
                break

    for vc in videoCaps:
        if not vc.cap.isOpened():
            print ("Could not open for reading " + vc.cap_name)
            sys.exit(2)

def arrange_windows(width, height):
    """
        Arranges the windows so they are not overlapping
        Also starts the display threads
    """
    spacer = 25
    cols = 0
    rows = 0

    # Arrange video windows
    for idx in range(len(videoCaps)):
        if(cols == WINDOW_COLUMNS):
            cols = 0
            rows += 1
        cv2.namedWindow(CAM_WINDOW_NAME_TEMPLATE.format("", idx), cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(CAM_WINDOW_NAME_TEMPLATE.format("", idx), (spacer + width) * cols, (spacer + height) * rows)
        cols += 1

    # Arrange statistics window
    if(cols == WINDOW_COLUMNS):
        cols = 0
        rows += 1
    cv2.namedWindow(STATS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(STATS_WINDOW_NAME, (spacer + width) * cols, (spacer + height) * rows)



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
    # Plugin initialization for specified device and load extensions library
    global rolling_log
    #defaultTarget = TARGET_DEVICE
    job_id = os.environ['PBS_JOBID']

    env_parser()
    args_parser()
    check_args()
    parse_conf_file(job_id)
    
    ie = IECore()
    if CPU_EXTENSION and 'CPU' in device:
        ie.add_extension(CPU_EXTENSION, "CPU")

    # Read IR
    print("Reading IR...")
    net = ie.read_network(model=model_xml, weights=model_bin)
    assert (len(net.input_info.keys()) == 1 or len(net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    for blob_name in net.input_info:
        if len(net.input_info[blob_name].input_data.shape) == 4:
            input_blob = blob_name
        elif len(net.input_info[blob_name].input_data.shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.input_info[blob_name].input_data.shape), blob_name))
    output_postprocessor = get_output_postprocessor(net)

    # Load the IR
    print("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=num_infer_requests, device_name=device)
    # Read and pre-process input image
    n, c, h, w = net.input_info[input_blob].input_data.shape

    del net

    minFPS = min([i.cap.get(cv2.CAP_PROP_FPS) for i in videoCaps])
    minlength = min([i.cap.get(cv2.CAP_PROP_FRAME_COUNT) for i in videoCaps])
    for vc in videoCaps:
        vc.rate = int(math.ceil(vc.length/minlength))
    waitTime = int(round(1000 / minFPS / len(videoCaps))) # wait time in ms between showing frames
    frames_sum = 0
    for vc in videoCaps:
        vc.init_vw(h, w, minFPS)
        frames_sum += vc.length
    statsWidth = w if w > 345 else 345
    statsHeight = h if h > (len(videoCaps) * 20 + 15) else (len(videoCaps) * 20 + 15)
    statsVideo = cv2.VideoWriter(os.path.join(output_dir,f'Statistics_{job_id}.mp4'), cv2.VideoWriter_fourcc(*"avc1"), minFPS, (statsWidth, statsHeight), True)    
    if not statsVideo.isOpened():
        print ("Couldn't open stats video for writing")
        sys.exit(4)

    # Read the labels file
    if labels_file:
        with open(labels_file, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # Init a rolling log to store events
    rolling_log_size = int((h - 15) / 20)
    rolling_log = collections.deque(maxlen=rolling_log_size)

    # Start with async mode enabled
    is_async_mode = True

    if not UI_OUTPUT:
        # Arrange windows so they are not overlapping
        #arrange_windows(w, h)
        print("To stop the execution press Esc button")

    no_more_data = False
    
    frame_count = 0
    progress_file_path = os.path.join(output_dir, f'i_progress_{job_id}.txt')
    infer_start_time = time.time()
    current_inference = 0
    previous_inference = 1 - num_infer_requests
    videoCapResult = {}
    #frames submitted to inference engine
    f_proc = 0
 
    
#Start while loop
    while True:
        # If all video captures are closed stop the loop
        if False not in [videoCap.closed for videoCap in videoCaps]:
            print("All videos completed")
            no_more_data = True
            break

        no_more_data = False

        # loop over all video captures
        for idx, videoCapInfer in enumerate(videoCaps):
            # read the next frame
            if not videoCapInfer.closed:
                 vfps = int(round(videoCapInfer.cap.get(cv2.CAP_PROP_FPS)))
                 for i in range(videoCapInfer.rate):
                     ret, frame = videoCapInfer.cap.read()
                     videoCapInfer.cur_frame_count += 1
                     # If the read failed close the program
                     if not ret:
                         videoCapInfer.closed = True
                         break
                     frame_count += 1
                     f_proc += 1

                     if videoCapInfer.closed:
                         print("Video {0} is done".format(idx))
                         print("Video has  {0} frames ".format(videoCapInfer.length))
                         break

                     # Copy the current frame for later use
                     videoCapInfer.cur_frame[current_inference] = frame.copy()
                     videoCapInfer.initial_w = int(videoCapInfer.cap.get(3))
                     videoCapInfer.initial_h = int(videoCapInfer.cap.get(4))
                     # Resize and change the data layout so it is compatible
                     in_frame = cv2.resize(frame, (w, h))
                     in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                     in_frame = in_frame.reshape((n, c, h, w))

                     inf_start = time.time()
                     if is_async_mode:
                         exec_net.start_async(request_id=current_inference, inputs={input_blob: in_frame})
                         # Async enabled and only one video capture
                         if(len(videoCaps) == 1):
                             videoCapResult = videoCapInfer
                         # Async enabled and more than one video capture
                         else:
                             # Get previous index
                             videoCapResult[current_inference] = videoCapInfer
                             videoCapInfer.start_time[current_inference] = time.time()
                     else:
                         # Async disabled
                         exec_net.start_async(request_id=current_inference, inputs={input_blob: in_frame})
                         videoCapResult = videoCapInfer

                     if previous_inference >= 0:
                        status = exec_net.requests[previous_inference].wait(-1)
                        if status == 0:
                            res = output_postprocessor(exec_net.requests[previous_inference].output_blobs)
                            vidcap = videoCapResult[previous_inference]
                            res_frame = vidcap.cur_frame[previous_inference]
                            end_time = time.time()
                            current_count = 0

                            infer_duration = end_time - vidcap.start_time[previous_inference]
                            applicationMetricWriter.send_inference_time(infer_duration*1000)                        
                            for obj in res:
                                class_id = int(obj[1])
                                # Draw only objects when probability more than specified threshold
                                if (obj[2] > PROB_THRESHOLD and
                                    vidcap.req_label in labels_map and
                                    labels_map.index(vidcap.req_label) == class_id - 1):
                                    current_count += 1
                                    xmin = int(obj[3] * vidcap.initial_w)
                                    ymin = int(obj[4] * vidcap.initial_h)
                                    xmax = int(obj[5] * vidcap.initial_w)
                                    ymax = int(obj[6] * vidcap.initial_h)
                                    # Draw box
                                    cv2.rectangle(res_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4, 16)

                            res_frame = cv2.resize(res_frame, (w, h))
                            
                            if vidcap.candidate_count is current_count:
                                vidcap.candidate_confidence += 1
                            else:
                                vidcap.candidate_confidence = 0
                                vidcap.candidate_count = current_count

                            if vidcap.candidate_confidence is FRAME_THRESHOLD:
                                vidcap.candidate_confidence = 0
                                if current_count > vidcap.last_correct_count:
                                    vidcap.total_count += current_count - vidcap.last_correct_count

                                if current_count is not vidcap.last_correct_count:
                                    if UI_OUTPUT:
                                        currtime = datetime.datetime.now().strftime("%H:%M:%S")
                                        fr = FrameInfo(vidcap.frames, current_count, currtime)
                                        vidcap.countAtFrame.append(fr)
                                    new_objects = current_count - vidcap.last_correct_count
                                    for _ in range(new_objects):
                                        str = "{} - {} detected on {}".format(time.strftime("%H:%M:%S"), vidcap.req_label, vidcap.cap_name)
                                        rolling_log.append(str)

                                vidcap.frames+=1
                                vidcap.last_correct_count = current_count
                            else:
                                vidcap.frames+=1

                            if not UI_OUTPUT:

                                # Add log text to each frame
                                log_message = "Async mode is on." if is_async_mode else \
                                              "Async mode is off."
                                cv2.putText(res_frame, log_message, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                log_message = "Total {} count: {}".format(vidcap.req_label, vidcap.total_count)
                                cv2.putText(res_frame, log_message, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                log_message = "Current {} count: {}".format(vidcap.req_label, vidcap.last_correct_count)
                                cv2.putText(res_frame, log_message, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.putText(res_frame, 'Infer wait: %0.3fs' % (infer_duration), (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                                # Display inferred frame and stats
                                stats = numpy.zeros((statsHeight, statsWidth, 1), dtype = 'uint8')
                                for i, log in enumerate(rolling_log):
                                    cv2.putText(stats, log, (10, i * 20 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                #cv2.imshow(STATS_WINDOW_NAME, stats)
                                if idx == 0:
                                    stats = cv2.cvtColor(stats, cv2.COLOR_GRAY2BGR)
                                    #Write
                                    statsVideo.write(stats)
                                cv2.putText(res_frame, 'FPS: %0.2fs' % (1 / (infer_duration)), (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                vidcap.video.write(res_frame)
                                del vidcap.cur_frame[previous_inference]

                      
                     if frame_count%10 == 0: 
                         progressUpdate(progress_file_path, time.time()-infer_start_time, frame_count, frames_sum) 
                     current_inference += 1
                     if current_inference >= num_infer_requests:
                         current_inference = 0

                     previous_inference += 1
                     if previous_inference >= num_infer_requests:
                         previous_inference = 0

            key = cv2.waitKey(1)
            # Esc key pressed
            if key == 27:
                cv2.destroyAllWindows()
                del exec_net
                del ie
                print("Finished")
                return
            # Tab key pressed
            if key == 9:
                is_async_mode = not is_async_mode
                print("Switched to {} mode".format("async" if is_async_mode else "sync"))

            # Loop video if LOOP_VIDEO = True and input isn't live from USB camera
            if LOOP_VIDEO and not videoCapInfer.is_cam:
                vfps = int(round(videoCapInfer.cap.get(cv2.CAP_PROP_FPS)))
                # If a video capture has ended restart it
                if (videoCapInfer.cur_frame_count > videoCapInfer.cap.get(cv2.CAP_PROP_FRAME_COUNT) - int(round(vfps / minFPS))):
                    videoCapInfer.cur_frame_count = 0
                    videoCapInfer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
            if no_more_data:
                break
#End of while loop--------------------

    progressUpdate(progress_file_path, time.time()-infer_start_time, frames_sum, frames_sum) 
    t2 = time.time()-infer_start_time
    print(f"total processed frames = {f_proc}")
    for videos in videoCaps:
        print(videos.closed)
        print("Frames processed {}".format(videos.cur_frame_count))
        print("Frames count {}".format(videos.length))
        videos.video.release()
        videos.cap.release()

    print("End loop")
    print("Total time {0}".format(t2))
    print("Total frame count {0}".format(frame_count))
    print("fps {0}".format(frame_count/t2))
    with open(os.path.join(output_dir, f'stats_{job_id}.txt'), 'w') as f:
        f.write('{} \n'.format(round(t2)))
        f.write('{} \n'.format(f_proc))

    applicationMetricWriter.send_application_metrics(model_xml, device)

if __name__ == '__main__':
    sys.exit(main() or 0)
