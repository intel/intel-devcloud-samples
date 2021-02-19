import argparse
import collections
import json
import logging
import os
import sys
import time

import tokenization
import numpy as np

from openvino.inference_engine import IECore
from run_factoid import write_predictions, read_squad_examples, convert_examples_to_features 

# Include devcloud demoutils
from qarpo.demoutils import *
import applicationMetricWriter
from qarpo.demoutils import progressUpdate

# Disable tensorflow logging when processing the data
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def is_safe_path(basedir, path):
    return os.path.abspath(path).startswith(basedir)

job_id = os.environ['PBS_JOBID']

# Set up logging
logging.basicConfig(format="[ %(asctime)s : %(levelname)s ] %(message)s", 
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", required=False,
                default='CPU', help="device type")
ap.add_argument("-o", "--output_dir", required=False,
                default='results', help="Location for output data")
args = vars(ap.parse_args())

device_type = args["device"]
output_dir = args["output_dir"]

basedir = os.getcwd()
if not is_safe_path(basedir, output_dir): 
    sys.stdout.write('Not allowed!\n') 
    sys.exit()
# BioBERT model parameters 
max_seq_length = 384
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

# Data files
input_file = "/data/BioBert/data-release/BioASQ-7b/test/Full-Abstract/BioASQ-test-factoid-7b-3.json"
vocab_file = os.path.join("/data/BioBert/BERT-pubmed-1000000-SQuAD", "vocab.txt")

data_features = []
all_results = []
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

# Define function to be used by the convert_examples_to_features function
def append_features(feature):
    data_features.append(feature)

# Use the functions from the pre-existing files to get the data 
logging.info("Loading data...")
eval_examples = read_squad_examples(input_file=input_file,
                                    is_training=False)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, 
                                       do_lower_case=False)
convert_examples_to_features(eval_examples, tokenizer,
                             max_seq_length, doc_stride,
                             max_query_length, False, 
                             append_features)

n = len(data_features)
logging.info("{} samples to process...".format(n))

if device_type != "TF":
    logging.info("Initializing IECore...")
    ie = IECore()

    logging.info("Reading IR...")
    net = ie.read_network(model = './ov/biobert.xml', 
                    weights = './ov/biobert.bin')

    logging.info("Generating Executable Network...")
    exec_net = ie.load_network(network=net, device_name=device_type)
    del net

    logging.info("Starting inference...")
    infer_start_time = time.time()
    for idx in range(n):
        data = {"input_ids": list(map(lambda x: x.input_ids, data_features[idx:idx+batch_size])),
                "input_mask": list(map(lambda x: x.input_mask, data_features[idx:idx+batch_size])),
                "segment_ids": list(map(lambda x: x.segment_ids, data_features[idx:idx+batch_size]))}

        unique_ids = list(map(lambda x: x.unique_id, data_features[idx:idx+batch_size]))
        
        inf_time = time.time()
        result = exec_net.infer(inputs=data)
        det_time = time.time()-inf_time
        applicationMetricWriter.send_inference_time(det_time*1000)                      


        # Process input and append to results list
        for i in range(batch_size):
            unique_id = unique_ids[i]
            sl = result["unstack/Squeeze_"][i,:]
            el = result["unstack/Split.1"][0,i,:]
            start_logits = [float(x) for x in sl.flat]
            end_logits = [float(x) for x in el.flat]
            all_results.append(RawResult(unique_id=unique_id, 
                                         start_logits=start_logits, 
                                         end_logits=end_logits))
            
        progressUpdate('./logs/' + str(job_id) + '.txt', 
                       time.time()-infer_start_time, idx+1, n)

    applicationMetricWriter.send_application_metrics('ov/biobert.xml', device_type)

else:
    from tensorflow.contrib import predictor

    logging.info("Loading TF Model...")
    predict_fn = predictor.from_saved_model('./tf_saved_model/' + os.listdir('./tf_saved_model/')[0])

    logging.info("Starting inference...")
    infer_start_time = time.time()
    for idx in range(n):
        data = {"input_ids": list(map(lambda x: x.input_ids, data_features[idx:idx+batch_size])),
                "input_mask": list(map(lambda x: x.input_mask, data_features[idx:idx+batch_size])),
                "segment_ids": list(map(lambda x: x.segment_ids, data_features[idx:idx+batch_size])),
                "unique_ids": list(map(lambda x: x.unique_id, data_features[idx:idx+batch_size]))}

        inf_time = time.time()
        result = predict_fn(data)
        det_time = time.time()-inf_time
        
        # Process input and append to results list
        for i in range(batch_size):
            unique_id = result["unique_ids"][i]
            sl = result["start_logits"][i,:]
            el = result["end_logits"][i,:]
            start_logits = [float(x) for x in sl.flat]
            end_logits = [float(x) for x in el.flat]
            all_results.append(RawResult(unique_id=unique_id, 
                                         start_logits=start_logits, 
                                         end_logits=end_logits))
            
        progressUpdate('./logs/' + str(job_id) + '.txt', 
                       time.time()-infer_start_time, idx+1, n)
        
total_time = time.time() - infer_start_time
logging.info("Inference took {} sec".format(total_time))

# Write performance stats to file
stats_file = output_dir + 'stats_'+str(job_id)+'.txt'
logging.info("Writing performance stats to {}".format(stats_file))
with open(stats_file, 'w') as f:
    f.write(str(((total_time/n)*1000))+'\n')
    f.write(str(n)+'\n')

# postprocessing filenames and directories
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

# Add flags so that the write_predictions function does not give
# "UnrecognizedFlagError: Unknown command line flag"
tf.app.flags.DEFINE_string('d', '', 'kernel')
tf.app.flags.DEFINE_string('o', '', 'kernel')

# Write the predictions to file using the function from run_factoid.py
logging.info("Writing predictions to {}".format(output_prediction_file))
write_predictions(eval_examples, data_features, all_results,
                  n_best_size, max_answer_length,
                  True, output_prediction_file, output_nbest_file, None)
