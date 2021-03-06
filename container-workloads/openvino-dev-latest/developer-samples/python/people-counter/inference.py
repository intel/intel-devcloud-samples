#!/usr/bin/env python3
"""
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

import os
import sys
import logging as log
import ngraph as ng
from openvino.inference_engine import IECore

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

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.ie = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, ie=None):
        """
         Loads a network and an image to the Inference Engine plugin.
        :param model: .xml file of pre trained model
        :param cpu_extension: extension for the CPU device
        :param device: Target device
        :param input_size: Number of input layers
        :param output_size: Number of output layers
        :param num_requests: Index of Infer request value. Limited to device capabilities.
        :param plugin: Plugin for specified device
        :return:  Shape of input layer
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device
        # and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(device))
        self.ie = IECore()
        if cpu_extension and 'CPU' in device:
            self.ie.add_extension(cpu_extension, "CPU")

        # Read IR
        log.info("Reading IR...")
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        
        #Ensure Model's layer's are supported by MKLDNN
        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            ng_function = ng.function_from_cnn(self.net)
            not_supported_layers = \
                    [node.get_friendly_name() for node in ng_function.get_ordered_ops() \
                    if node.get_friendly_name() not in supported_layers]
            
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(args.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        assert (len(self.net.input_info.keys()) == 1 or len(self.net.input_info.keys()) == 2),\
        "Sample supports topologies only with 1 or 2 inputs"

        for blob_name in self.net.input_info:
            if len(self.net.input_info[blob_name].input_data.shape) == 4:
                self.input_blob = blob_name
            elif len(self.net.input_info[blob_name].input_data.shape) == 2:
                self.img_info_input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(self.net.input_info[blob_name].input_data.shape), blob_name))

        self.output_postprocessor = self.get_output_postprocessor(self.net)
        
        log.info("Loading IR to the plugin...")
        if num_requests == 0:
            self.net_plugin = self.ie.load_network(network=self.net, device_name=device)
        else:
            self.net_plugin = self.ie.load_network(network=self.net, device_name=device, num_requests=num_requests)
            
        return self.get_input_shape()
       
    def get_input_shape(self):
        """
        Gives the shape of the input layer of the network.
        :return: None
        """
        return self.net.input_info[self.input_blob].input_data.shape

    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.net_plugin.requests[request_id].get_perf_counts()
        return perf_count

    def exec_net(self, request_id, frame):
        """
        Starts asynchronous inference for specified request.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param frame: Input image
        :return: Instance of Executable Network class
        """
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        return self.net_plugin

    def wait(self, request_id):
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        wait_process = self.net_plugin.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.output_postprocessor(self.net_plugin.requests[request_id].output_blobs)
        return res
    
    def get_output_postprocessor(self, net, bboxes='bboxes', labels='labels', scores='scores'):
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

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.net_plugin
        del self.ie
        del self.net
