import os
from time import time
from warnings import simplefilter 
simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

import keras
import numpy as np
import scipy.stats as sst
import sklearn.metrics as skm
from keras.backend.tensorflow_backend import tf
from tqdm import tqdm

import load

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = "/data/ecg/0.427-0.863-020-0.290-0.899.hdf5"
data_csv = "./data/reference.csv"

print("Loading Dataset")
ecgs, labels = load.load_dataset(data_csv)

print("Loading Model")
model = keras.models.load_model(model_path)

print("Starting Inference")
probs = []
total_time = 0
for x in tqdm(ecgs):
    x = load.process_x(x)
    start_time = time()
    probs.append(model.predict(x))
    total_time += (time() - start_time)
    
print("Keras took {} sec for inference".format(total_time))

# The class distribution of the overall dataset
prior = [[[0.15448743, 0.66301941, 0.34596848, 0.09691286]]]

# Determine the predicted class from the most commonly predicted class 
preds = []
for p in probs:
    preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])

# Generate a report with the precision, recall, and f-1 scores for each of the classes
report = skm.classification_report(labels, preds, target_names=['A','N','O','~'], digits=3)
scores = skm.precision_recall_fscore_support(labels, preds, average=None)

print(report)
print ("CINC Average {:3f}".format(np.mean(scores[2][:3])))
