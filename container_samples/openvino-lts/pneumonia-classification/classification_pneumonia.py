from __future__ import print_function
import sys
import os,glob
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IECore
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import struct
import time

from utils import load_img, img_to_array, resize_image
from pathlib import Path
from qarpo.demoutils import simpleProgressUpdate
#import applicationMetricWriter


def float16_conversion(n):
     w1=struct.pack('H', int(np.binary_repr(int(n),width=16), 2))
     w=np.frombuffer(w1,dtype=np.float16)[0]
     return w 
    
def float16_conversion_array(n_array):
    c=n_array.flatten()
    c1=np.array([float16_conversion(xi) for xi in c]).reshape(n_array.shape)
    return c1

def class_activation_map_openvino(res, convb, fc, net, fp16):
    res_bn = res[convb]
    conv_outputs=res_bn[0,:,:,:]
    # retrieve layer weights
    weights_fc=net.layers.get(fc).blobs["weights"]
    # initialize CAM array
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:])
    # perform weighted sum
    for i, w in enumerate(weights_fc):
        conv_outputs1=conv_outputs[i, :, :]
        if fp16:
            w=float16_conversion(w)
            conv_outputs1=float16_conversion_array(conv_outputs[i, :, :])
        cam += w * conv_outputs1
    return cam

def read_image(path):
    image_original = load_img(path, color_mode="rgb")
    img= resize_image(image_original, target_size=(224, 224))
    x = img_to_array(img, data_format='channels_first')
    return [x,image_original]

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str, nargs="+")
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=20, type=int)
    parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)

    return parser


def main():
    
    colormap='viridis'
    #job_id = os.environ['PBS_JOBID']
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    device=args.device
    
    fp16=True

    if "CPU" in device:
        fp16=False

    # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()

    # Read IR
    net = ie.read_network(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"

    bn = "relu_1/Relu"
    print(bn)
    # add the last convolutional layer as output 
    net.add_outputs(bn)
    """
    Note: Layer name should be "MatMul", however it appears as "BiasAdd/Add" in
          the IR from Model Optimizer in OpenVINO 2020.1.  
          Below contains a workaround until corrected in the next release.
    """
    #fc="predictions_1/MatMul"
    fc="predictions_1/BiasAdd/Add"

    # name of the inputs and outputs
    input_blob = next(iter(net.inputs))
    out_blob = "predictions_1/Sigmoid"

    net.batch_size = 1

    exec_net = ie.load_network(network=net, device_name=args.device)

    n,c,h,w=net.inputs[input_blob].shape
    # Use strip in case of " or ' present which might be used to avoid shell globing in scripts
    files=glob.glob(args.input[0].strip("\"'")) 
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    f=open(os.path.join(args.output_dir, f'result.txt'), 'w')
    f1=open(os.path.join(args.output_dir, f'stats.txt'), 'w') 
    progress_file_path = os.path.join(args.output_dir, f'i_progress.txt')
    time_images=[]
    tstart=time.time()
    for index_f, file in enumerate(files):
        [image1,image]= read_image(file)
        t0 = time.time()
        for i in range(args.number_iter):
            inf_time = time.time()
            res = exec_net.infer(inputs={input_blob: image1})
            det_time = time.time() - inf_time
            #applicationMetricWriter.send_inference_time(det_time*1000)   
        infer_time = (time.time() - t0)*1000
        log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
        if args.perf_counts:
            perf_counts = exec_net.requests[0].get_perf_counts()
            log.info("Performance counters:")
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
            for layer, stats in perf_counts.items():
                print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                                  stats['status'], stats['real_time']))
        res_pb = res[out_blob]
        probs=res_pb[0][0]
        print("Probability of having disease= "+str(probs)+", performed in " + str(np.average(np.asarray(infer_time))) +" ms")
        
        # Class Activation Map    
        t0 = time.time()
        cam=class_activation_map_openvino(res, bn, fc , net, fp16)
        cam_time=(time.time() - t0) * 1000
        print("Time for CAM: {} ms".format(cam_time))


        fig,ax = plt.subplots(1,2)
        # Visualize the CAM heatmap
        np.seterr(divide='ignore', invalid='ignore')
        cam = (cam - np.min(cam))/(np.max(cam)-np.min(cam))
        im=ax[0].imshow(cam, cmap=colormap)
        ax[0].axis('off')
        plt.colorbar(im,ax=ax[0],fraction=0.046, pad=0.04)

        # Visualize the CAM overlaid over the X-ray image 
        colormap_val=cm.get_cmap(colormap)  
        imss=np.uint8(colormap_val(cam)*255)
        im = Image.fromarray(imss)
        width, height = image.size
        cam1=resize_image(im, (height,width))
        heatmap = np.asarray(cam1)
        img1 = heatmap [:,:,:3] * 0.3 + image
        ax[1].imshow(np.uint16(img1))
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.savefig(os.path.join(args.output_dir, f'result_{index_f}.png'), bbox_inches='tight', pad_inches=0,dpi=300)
       
        avg_time = round((infer_time/args.number_iter), 1)
        
                    #f.write(res + "\n Inference performed in " + str(np.average(np.asarray(infer_time))) + "ms") 
        f.write("Pneumonia probability: "+ str(probs) + ", Inference performed in " + str(avg_time) + "ms, Input file: "+file+" \n") 
        time_images.append(avg_time)
        simpleProgressUpdate(progress_file_path,index_f* avg_time , (len(files)-1)* avg_time) 
    total_time = np.sum(np.asarray(time_images))/1000
    f1.write(str(total_time)+'\n')
    f1.write(str(len(time_images))+'\n')
    #applicationMetricWriter.send_application_metrics(model_xml, device)

if __name__ == '__main__':
    sys.exit(main() or 0)
