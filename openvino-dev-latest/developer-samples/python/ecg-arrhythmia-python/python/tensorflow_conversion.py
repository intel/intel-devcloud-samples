import os
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from keras.models import load_model
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from tensorflow.python.framework import graph_io

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.clear_session()
K.set_learning_phase(0)

input_model = '/data/ecg/0.427-0.863-020-0.290-0.899.hdf5'
output_model = 'models/output_graph.pb'
num_output = 1 

print("Loading Model")
model = load_model(input_model)
print(model.summary())

predictions = [None] * num_output
predrediction_node_names = [None] * num_output

for i in range(num_output):
    predrediction_node_names[i] = 'output_node' + str(i)
    predictions[i] = tf.identity(model.outputs[i], 
    name=predrediction_node_names[i])

sess = K.get_session()

constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), predrediction_node_names)
infer_graph = tf.compat.v1.graph_util.remove_training_nodes(constant_graph) 

print("Saving converted model to {}".format(output_model))
graph_io.write_graph(infer_graph, '.', output_model, as_text=False)
