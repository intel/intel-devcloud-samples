import os
import logging as log
#assert 'computer_vision_sdk' in os.environ['PYTHONPATH']

from PIL import Image
import numpy as np
import cv2
import sys
import os
from argparse import ArgumentParser
from qarpo.demoutils import *
import applicationMetricWriter
from time import time
import glob
import ngraph as ng

try:
    #from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IECore
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input_path", help="Path to a folder with images or path to an image files", required=False,
                        type=str)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=1000, type=int)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("--num_threads", default=88, type=int)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")
    #parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
    #                    default=None, type=str)

    parser.add_argument("-p", "--out_dir", help="Optional. The path where result files and logs will be stored",
                      required=False, default="./results", type=str)
    parser.add_argument("-o", "--out_prefix", 
                      help="Optional. The file name prefix in the output_directory where results will be stored", 
                      required=False, default="out_", type=str)
    parser.add_argument("-g", "--log_prefix", 
                      help="Optional. The file name prefix in the output directory for log files",
                      required=False, default="log_", type=str)

    return parser

def main():
    # Run inference
    #{'ZIL131': 8, '2S1': 0, 'BTR70': 3, 'BTR_60': 4, 'T72': 7, 'T62': 6, 'D7': 5, 'BMP2': 1, 'BRDM_2': 2, 'ZSU_23_4': 9}
    class_labels = np.array(['2S1', 'BMP2', 'BRDM_2', 'BTR70', 'BTR_60', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4'])

    job_id = os.getenv("PBS_JOBID")    
    # Plugin initialization for specified device and load extensions library if specified.
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Set up logging to a file
    logpath = os.path.join(os.path.dirname(__file__), 
                           "{}/{}_{}.log".format(args.out_dir, args.log_prefix, job_id))
    log.basicConfig(level=log.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    filename=logpath,
                    filemode="w" )
    try:
        job_id = os.environ['PBS_JOBID']
        infer_file = os.path.join(args.out_dir,'i_progress_'+str(job_id)+'.txt')
    except Exception as e:
        log.warning(e)
        log.warning("job_id: {}".format(job_id))
    
    # Setup additional logging to console
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    formatter = log.Formatter("[ %(levelname)s ] %(message)s")
    console.setFormatter(formatter)
    log.getLogger("").addHandler(console)
    
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    ie = IECore()
   
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading plugins for {} device...".format(args.device))
        ie.add_extension(args.cpu_extension, "CPU")

    # Read IR
    log.info("Reading IR...")
    net = ie.read_network(model=model_xml, weights=model_bin)
    #net = IENetwork(model=model_xml, weights=model_bin)
    
    #Ensure Model's layer's are supported by MKLDNN
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        ng_function = ng.function_from_cnn(net)
        not_supported_layers = \
                [node.get_friendly_name() for node in ng_function.get_ordered_ops() \
                if node.get_friendly_name() not in supported_layers]

        if len(not_supported_layers) != 0:
            log.warning("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.warning("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
            
    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    
    # Load network to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    
    
    CLASSES = 10
    
    image_files = glob.glob(os.path.join(args.input_path, "*", "*.jpg"))
    image_files.extend(glob.glob(os.path.join(args.input_path, "*", "*.JPG")))
    labels = [fname.split("/")[-2] for fname in image_files]

    infer_time = []
    correct = 0; error = 0
    t0 = time()
    n, c, h, w = net.input_info[input_blob].input_data.shape
    log.info("Batch size is {}".format(n))    
    log.info("Starting inference ({} iterations)".format(len(image_files)))
    for i in range(0, len(image_files), n):
        images = np.ndarray(shape=(n, c, h, w))
        for j in range(n):
            input = image_files[i*n + j]
            image = cv2.imread(input, 0) # Read image as greyscale
            if image.shape[-2:] != (h, w):
                log.warning("Image {} is resized from {} to {}".format(input, image.shape[-2:], (h, w)))
                image = cv2.resize(image, (w, h))

            # Normalize to keep data between 0 - 1
            image = (np.array(image) - 0) / 255.0

            # Change data layout from HWC to CHW
            image = image.reshape((1, 1, h, w))    
            images[j] = image
        try:
            t0P = time()
            result = exec_net.infer(inputs={input_blob: images})
            infer_time.append((time()-t0P)*1000)
            try:
                applicationMetricWriter.send_inference_time((time()-t0P)*1000)
            except Exception as e:
                log.warning(e)
            if i%10 == 0: 
                try:
                    progressUpdate(infer_file, time()-t0, i+1, len(image_files))
                except Exception as e:
                    log.warning(e)

            # Access the results and get the index of the highest confidence score
            result = result[out_blob]

            # Predicted class index.
            class_num = np.argmax(result)
            for j in range(n):
                class_label = class_labels[class_num]
                true_label = labels[i*n + j]
             
            
                if class_label == true_label:
                    correct += 1
                else:
                    error += 1
                    #log.info("incorrect image: {} - predicted {} - ground truth {}".format(input, class_label, true_label))
            
            
        except Exception as e:
            log.warning("Exception: {}".format(e))
    log.info("Average running time of infer: {} ms".format(np.average(np.asarray(infer_time)))) 
    log.info('Correct: ' + str(correct))
    log.info('Error: ' + str(error))
    accuracy = float(correct) / float(len(image_files))
    log.info('Accuracy ' + str(accuracy *  100))

    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    if args.perf_counts:
        perf_counts = exec_net.requests[0].get_perf_counts()
        log.info("Performance counters:")
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
        for layer, stats in perf_counts.items():
            log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))
    
    t1 = (time() - t0)*1000   
    log.info("Total running time: {} ms".format((np.asarray(t1))))  
    
    with open(os.path.join(args.out_dir, 'result_'+str(job_id)+'.txt'), 'w') as f:
                f.write('Accuracy: ' + str(accuracy *  100) + "% \nAverage Inference Time: " + str(np.average(np.asarray(infer_time))) + "ms")  
    with open(os.path.join(args.out_dir, 'stats_'+str(job_id)+'.txt'), 'w') as f:
                f.write(str(np.average(np.asarray(infer_time))/1000.0)+'\n')
                f.write(str(1)+'\n')
                
    applicationMetricWriter.send_application_metrics(model_xml, args.device)


if __name__ == '__main__':
    sys.exit(main() or 0) 
