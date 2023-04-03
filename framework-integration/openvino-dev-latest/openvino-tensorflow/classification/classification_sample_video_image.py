# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

# Modified from TensorFlow example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
#https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_classification.ipynb
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
# Enable these variables for runtime inference optimizations
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"
os.environ[
    "TF_ENABLE_ONEDNN_OPTS"] = "1"
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import openvino_tensorflow as ovtf
from PIL import Image
import time
import cv2

import sys
sys.path.append("../")
from common.utils import get_input_mode

def preprocess_image(frame,
                     input_height=299,
                     input_width=299,
                     input_mean=0,
                     input_std=255):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resized_image = image.resize((input_height, input_width))
    resized_image = np.asarray(resized_image, np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    if(backend_name == "VAD-M"):
        return normalized_image
    else:
        result = np.expand_dims(normalized_image, 0)
        return result

def load_labels(label_file):
    label = []
    assert os.path.exists(label_file), "Could not find label file path"
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def run_infer(model, label_file, input_file, input_height, input_width, input_mean, input_std, backend_name, filename, output_filename):
    
    #Load the labels
    cap = None
    video_writer = None
    images = []
    if label_file:
        labels = load_labels(label_file)
    input_mode = get_input_mode(input_file)
    if input_mode == 'video':
        cap = cv2.VideoCapture(input_file)
        video_writer = cv2.VideoWriter()
        output_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )))
        video_writer.open(output_filename, cv2.VideoWriter_fourcc(*'avc1'), 20.0, output_resolution)
    elif input_mode == 'image':
        images = [input_file]
    elif input_mode == 'directory':
        images = [os.path.join(input_file, i) for i in os.listdir(input_file)]
    else:
        raise Exception(
            "Invalid input. Path to an image or video or directory of images. Use 0 for using camera as input."
        )
    images_len = len(images)
   
    # Preprocess image and run inference
    image_id = -1
    while True:
        image_id += 1
        if input_mode == 'video':
            if cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    pass
                else:
                    break
            else:
                break
        if input_mode in ['image', 'directory']:
            if image_id < images_len:
                frame = cv2.imread(images[image_id])
            else:
                break

        if(backend_name == "VAD-M"):
            xlist = []
            for i in range(1, 9):
                x = tf.convert_to_tensor(preprocess_image(frame, input_height=input_height, input_width=input_width))
                xlist.append(x)
            t = np.stack(xlist)
        else:
            t = tf.convert_to_tensor(preprocess_image(frame, input_height=input_height, input_width=input_width))

        # Warmup
        if image_id == 0:
            results = model(t)

        # run
        start = time.time()
        results = model(t)
        elapsed = time.time() - start
        result_file_name = "/mount_folder/" +"performance.txt"
        if(flag_enable == "openvino"): 
            f = open(result_file_name, "w")
            f.write('Openvino Integration with Tensorflow \n')    
        else:
            f = open(result_file_name, "a")
            f.write('Stock Tensorflow \n')
        fps = 1 / elapsed
        print('Inference time in ms: %.2f' % (elapsed * 1000))
        f.write('Throughput: {:.3g} FPS \n'.format(fps))
        f.write('Latency: {:.3f} ms\n'.format(1000*elapsed))
        f.close()
        results = tf.nn.softmax(results).numpy()
        
        if label_file:
            cv2.putText(frame,
                        'Inference Running on : {0}'.format(backend_name),
                        (30, 50), font, font_size, color, font_thickness)
            cv2.putText(
                frame, 'FPS : {0} | Inference Time : {1}ms'.format(
                    int(fps), round((elapsed * 1000), 2)), (30, 80), font,
                font_size, color, font_thickness)
            top_5 = tf.argsort(
                results, axis=-1, direction="DESCENDING")[0][:5].numpy()
            c = 130
            result_file_name = "/mount_folder/" +"performance.txt"
            f = open(result_file_name, "a")
            for i, item in enumerate(top_5):
                cv2.putText(
                    frame, '{0} : {1}'.format(labels[item],
                                              results[0][top_5][i]), (30, c),
                    font, font_size, color, font_thickness)
                
                st = ""
                print(labels[item], results[0][top_5][i])
                st = st + labels[item]+" \t"+ str(results[0][top_5][i])+" \n"
                c += 30
            f.write(st)
            f.close()
        else:
            print("No label file provided. Cannot print classification results")
        if not args.no_show:
           # cv2.imshow("results", frame)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        if input_mode == 'video':
            video_writer.write(frame)
    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_file = tf.keras.utils.get_file(
        'grace_hopper.jpg',
        "https://www.tensorflow.org/images/grace_hopper.jpg")
    model_file = ""
    label_file = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    backend_name = "CPU"
    
    flag_enable = "openvino" # please edit for oneDNN and native TensorFlow
    output_filename = "output_detection.mp4" # edit the name accordingly
    
    # overlay parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = .6
    color = (0, 0, 0)
    font_thickness = 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="Optional. Path to model to be executed.")
    parser.add_argument(
        "--labels", help="Optional. Path to labels mapping file.")
    parser.add_argument(
        "--input",
        help=
        "Optional. The input to be processed. Path to an image or video or directory of images. Use 0 for using camera as input"
    )
    parser.add_argument(
        "--input_height",
        type=int,
        help="Optional. Specify input height value.")
    parser.add_argument(
        "--input_width", type=int, help="Optional. Specify input width value.")
    parser.add_argument(
        "--input_mean", type=int, help="Optional. Specify input mean value.")
    parser.add_argument(
        "--input_std", type=int, help="Optional. Specify input std value.")
    parser.add_argument(
        "--backend",
        help="Optional. Specify the target device to infer on; "
        "CPU, GPU, MYRIAD or VAD-M is acceptable. Default value is CPU.")
    parser.add_argument(
        "--no_show", help="Optional. Don't show output.", action='store_true')
    parser.add_argument(
        "--disable_ovtf",
        help="Optional. Disable openvino_tensorflow pass and run on stock TF.",
        action='store_true')
    parser.add_argument("-f", "--flag", help="disable backend")
    parser.add_argument("-of","--output_file", help="output video filename")

    args = parser.parse_args()

    if args.model:
        model_file = args.model
        if args.labels:
            label_file = args.labels
        else:
            label_file = None
    if args.input:
        input_file = args.input
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.backend:
        backend_name = args.backend

    if model_file == "":
        model = hub.load(
            "https://tfhub.dev/google/imagenet/inception_v3/classification/4")
    else:
        model = tf.saved_model.load(model_file)

    #Print list of available backends
    if args.output_file:
        output_filename = args.output_file
    flag_enable = args.flag
    filename = ""
    if(flag_enable == "native"):
        print('StockTensorflow')
        filename = "native"
        ovtf.disable()
    elif(flag_enable == "oneDNN"):
        print('oneDNN optimized')
        filename = "oneDNN"
        ovtf.disable()
        os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
    elif(flag_enable == "openvino"):
        filename = "openvino"
        print('Openvino Integration With Tensorflow')
        print('Available Backends:')
        backends_list = ovtf.list_backends()
        for backend in backends_list:
            print(backend)
        os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
        ovtf.set_backend(backend_name)
    else:
        raise AssertionError("flag_enable string not supported")
    
    run_infer(model, label_file, input_file, input_height, input_width, input_mean, input_std, backend_name, filename, output_filename)
