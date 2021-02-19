#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np
import logging as log
import time
from openvino.inference_engine import IECore
from local_utils import log_utils, data_utils
from local_utils.config_utils import load_config
import os.path as ops
from easydict import EasyDict
from qarpo.demoutils import *
import applicationMetricWriter
import ngraph as ng

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str, nargs="+")
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=1000, type=int)
    parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)
    parser.add_argument("-nir", "--num_infer_req", help="Number of Async inference request", type=int, default=2)

    return parser


class SingleOutputPostprocessor:
    def __init__(self, output_layer):
        self.output_layer = output_layer

    def __call__(self, outputs):
        return outputs[self.output_layer].buffer


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
   
    job_id = os.environ['PBS_JOBID']
    codec = data_utils.TextFeatureIO(char_dict_path='Config/char_dict.json',ord_map_dict_path=r'Config/ord_map.json')

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    #Ensure Model's layer's are supported by MKLDNN
    if args.device == "CPU":
        supported_layers = ie.query_network(net, args.device)
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

    output_postprocessor = get_output_postprocessor(net)

    job_id = str(os.environ['PBS_JOBID'])
    infer_file = os.path.join(args.output_dir, 'i_progress_'+str(job_id)+'.txt')
    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=args.num_infer_req, device_name=args.device)
    del net

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    if args.number_iter > args.num_infer_req:
        num_infer_req = args.num_infer_req
    else:
        num_infer_req = args.number_iter
    print ("Number of Requested Infer:", num_infer_req)

    infer_time = []
    t0 = time.time()
    print(args.number_iter)

    cur_infer_id = 0
    pre_infer_id = 1 - num_infer_req
    iter_count = 0
    iter_out_count = 1
    infer_requests = exec_net.requests

    t0 = time.time()
    while True:
        if (iter_count <= args.number_iter):
            req_time = time.time()
            exec_net.start_async(request_id = cur_infer_id, inputs={input_blob: images})
            iter_count += 1

        if pre_infer_id >= 0:
            inf_time = time.time()
            infer_requests[pre_infer_id].wait()
            res = output_postprocessor(exec_net.requests[pre_infer_id].output_blobs)
            det_time = time.time() - inf_time
            applicationMetricWriter.send_inference_time(det_time*1000)
            print ("Det time", det_time)

            if (iter_out_count % 10 == 0) or (iter_out_count == args.number_iter):
                progressUpdate(infer_file, time.time()-t0, iter_out_count, args.number_iter)
            iter_out_count += 1
            if (iter_out_count > args.number_iter):
                print ("Deleting the iter_out_count", iter_out_count)
                print ("Deleting Variable iter_count at", iter_count)
                del (iter_count)
                del (iter_out_count)
                break

        cur_infer_id += 1
        pre_infer_id += 1

        if cur_infer_id >= num_infer_req:
            cur_infer_id = 0
        if pre_infer_id >= num_infer_req:
            pre_infer_id = 0

        #infer_time.append((time()-t0)*1000)
    t1 = (time.time() - t0)*1000
    
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    if args.perf_counts:
        perf_counts = requests[pre_infer_id].get_perf_counts()
        log.info("Performance counters:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
        for layer, stats in perf_counts.items():
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))

    # Processing output blob
    log.info("Processing output blob")
    #res = res[out_blob]

    preds = res.argmax(2)
    preds = preds.transpose(1, 0)
    preds = np.ascontiguousarray(preds, dtype=np.int8).view(dtype=np.int8)
    values=codec.writer.ordtochar( preds[0].tolist())
    values=[v for i, v in enumerate(values) if i == 0 or v != values[i-1]]
    values = [x for x in values if x != ' ']
    res=''.join(values)
    print("The result is : " + res)
    
    #progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')
    avg_time = round((t1/args.number_iter), 1)
    with open(os.path.join(args.output_dir, f'result_{job_id}.txt'), 'w') as f:
                #f.write(res + "\n Inference performed in " + str(np.average(np.asarray(infer_time))) + "ms") 
                f.write(res + "\n Inference performed in " + str(avg_time) + "ms") 
    with open(os.path.join(args.output_dir, f'stats_{job_id}.txt'), 'w') as f:
        f.write('{} \n'.format(round(avg_time)))
        f.write('{} \n'.format(args.number_iter))
    applicationMetricWriter.send_application_metrics(model_xml, args.device)

if __name__ == '__main__':
    sys.exit(main() or 0)
