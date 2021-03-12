import argparse
import numpy as np
import logging as log
import os
import sys
from time import time
from qarpo.demoutils import progressUpdate
import applicationMetricWriter

import scipy.stats as sst
import sklearn.metrics as skm

import load

from openvino.inference_engine import IECore

job_id = os.environ['PBS_JOBID']

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", required=False,
                default='CPU', help="device type")
ap.add_argument("-o", "--output_dir", required=False,
                default='results/', help="Location for output data")
args = vars(ap.parse_args())

# Arguments
device_type = args['device']
output_dir = args['output_dir']

data_csv = "./data/reference.csv"

#checking list of python packages TODO: if not needed and for debug only please comment out

import pkg_resources

dists = [d for d in pkg_resources.working_set]
log.info(dists)

log.info("Loading Dataset")
ecgs, labels = load.load_dataset(data_csv, progress_bar=False)

# Load network and add CPU extension if device is CPU
ie = IECore()
net = ie.read_network(model = './models/output_graph.xml', weights = './models/output_graph.bin')
exec_net = ie.load_network(network=net, device_name=device_type)

prior = [[[0.15448743, 0.66301941, 0.34596848, 0.09691286]]]

infer_time_start = time()
 
probs_total = []
total_time = 0
count = 0
sample_count = len(ecgs)

for x in ecgs:
    x = load.process_x(x)
    start_time = time()
    res = exec_net.infer(inputs={"inputs": x})
    inf_time = (time() - start_time)
    applicationMetricWriter.send_inference_time(inf_time*1000)
    total_time += inf_time 
    probs = res["time_distributed_1/Reshape_1/Softmax"]
    probs_total.append(probs)

    count += 1
    progressUpdate('./logs/' + str(job_id) + '.txt', time()-infer_time_start, count, sample_count)

    
log.info("OpenVINO took {} sec for inference".format(total_time))
with open(os.path.join(os.getcwd(), output_dir + 'stats_'+str(job_id)+'.txt'), 'w') as f:
    f.write(str(round(((total_time/sample_count)*1000), 1))+'\n')
    f.write(str(sample_count)+'\n')

applicationMetricWriter.send_application_metrics('./models/output_graph.xml', device_type)


# Determine the predicted class from the most commonly predicted class
preds = []
for p in probs_total:
    preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
    
# Generate a report with the precision, recall, and f-1 scores for each of the classes
report = skm.classification_report(labels, preds, target_names=['A','N','O','~'], digits=3)
scores = skm.precision_recall_fscore_support(labels, preds, average=None)
    
log.info(report)
log.info ("CINC Average {:3f}".format(np.mean(scores[2][:3])))
