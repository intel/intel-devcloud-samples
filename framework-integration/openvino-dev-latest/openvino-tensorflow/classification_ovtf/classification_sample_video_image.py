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
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import openvino_tensorflow as ovtf
import time
import cv2


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    assert os.path.exists(model_file), "Could not find model path"
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def read_tensor_from_video_file(frame,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    resized = cv2.resize(frame, (input_height, input_width))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized_image = img.astype(np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    result = np.expand_dims(normalized_image, 0)
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def run_video_infer(model_file, input_layer, output_layer,label_file,input_file,input_height,input_width, input_mean,input_std, backend_name,flag_enable):
    
    # Read input video file
    cap = cv2.VideoCapture(input_file)
    frame_res = cap.read()
    #video_writer = cv2.VideoWriter()
    output_resolution = (input_height, input_width)
    
    # Initialize session and run
    config = tf.compat.v1.ConfigProto()
    video_writer = cv2.VideoWriter()
    output_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )))
    print(output_resolution)
    print("video")
    print(output_resolution)
    video_writer.open("out_demo.mp4", cv2.VideoWriter_fourcc(*'MJPG'),
                                                         20.0, output_resolution)
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        while cap.isOpened():
            ret, frame = cap.read()
           
            if ret is True:
                t = read_tensor_from_video_file(
                    frame,
                    input_height=input_height,
                    input_width=input_width,
                    input_mean=input_mean,
                    input_std=input_std)

                # Run
                start = time.time()
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
                elapsed = time.time() - start
                fps = 1 / elapsed
                print('Inference time in ms: %f' % (elapsed * 1000))
                results = np.squeeze(results)
                
                # print labels
                if label_file:
                    cv2.putText(
                        frame,
                        'Inference Running on : {0}'.format(backend_name),
                        (30, 50), font, font_size, color, font_thickness)
                    cv2.putText(
                        frame, 'FPS : {0} | Inference Time : {1}ms'.format(
                            int(fps), round((elapsed * 1000), 2)), (30, 80),
                        font, font_size, color, font_thickness)
                    top_k = results.argsort()[-5:][::-1]
                    c = 130
                    for i in top_k:

                        cv2.putText(frame, '{0} : {1}'.format(
                            labels[i], results[i]), (30, c), font, font_size,
                                    color, font_thickness)
                        print(labels[i], results[i])
                        c += 30
                else:
                    print(
                        "No label file provided. Cannot print classification results"
                    )
                name = "output" + '.jpg'

                #cv2.imwrite(name,frame)
               # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                
               # cv2.imshow("results", frame)
                #cv2.imwrite("output_class.jpg",frame)
                video_writer.write(frame)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                print("Completed")
                break
    video_writer.release()
   
    cap.release()
    cv2.destroyAllWindows()
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    assert os.path.exists(file_name), "Could not find image file path"
    image = cv2.imread(file_name)
    resized = cv2.resize(image, (input_height, input_width))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized_image = img.astype(np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    result = np.expand_dims(normalized_image, 0)

    return result

def run_image_infer(model_file, input_layer, output_layer,label_file,file_name,input_height,input_width, input_mean,input_std, backend_name,flag_enable):
    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        t = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        # Warmup
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})

        # Run
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        elapsed = time.time() - start
        print('Inference time in ms: %f' % (elapsed * 1000))
       # result_file_name = "results/" + "tensorflow_"+ str(flag_enable) + backend_name + ".txt"
       # f = open(result_file_name, "w")
        #f.write(str((elapsed * 1000)))
        #f.close()
    results = np.squeeze(results)

    # print labels
    if label_file:
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
            print(labels[i], results[i])
    else:
        print("No label file provided. Cannot print classification results")


if __name__ == "__main__":
    input_file = "examples/data/people-detection.mp4"
    model_file = "examples/data/inception_v3_2016_08_28_frozen.pb"
    label_file = "examples/data/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    backend_name = "CPU"

    # overlay parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = .6
    color = (0, 0, 0)
    font_thickness = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--graph", help="graph/model to be executed")
    parser.add_argument("-i","--input_layer", help="name of input layer")
    parser.add_argument("-o","--output_layer", help="name of output layer")
    parser.add_argument("-l","--labels", help="name of file containing labels")
    parser.add_argument(
        "-ip","--input",
        help="input (0 - for camera / absolute video file path) to be processed"
    )
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("-d","--backend", help="backend option. Default is CPU")
    parser.add_argument("-f", "--flag", help="disable backend")
    parser.add_argument("-it","--input_type", help="input type either video or image")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
        if not args.input_layer:
            raise Exception("Specify input layer for this network")
        else:
            input_layer = args.input_layer
        if not args.output_layer:
            raise Exception("Specify output layer for this network")
        else:
            output_layer = args.output_layer
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

    graph = load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    #Print list of available backends
    
    flag_enable = args.flag
    flag_input = args.input_type
    filename = ""
    if(flag_enable == "tf"):
        print('StockTensorflow')
        filename = "native"
        ovtf.disable()
    elif(flag_enable == "oneDNN"):
        filename = "oneDNN"
        print('oneDNN optimized')
        ovtf.disable()
        os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
    else:
        print('Available Backends:')
        filename = "openvino"
        backends_list = ovtf.list_backends()
        for backend in backends_list:
            print(backend)
        ovtf.set_backend(backend_name)
    if label_file:
        labels = load_labels(label_file)
    if(flag_input == "video"):
        run_video_infer(model_file, input_layer, output_layer,label_file,input_file,input_height,input_width, input_mean,input_std, backend_name,flag_enable)
    else:
        run_image_infer(model_file, input_layer, output_layer,label_file,input_file,input_height,input_width, input_mean,input_std, backend_name,filename)
    #Load the labels
    

