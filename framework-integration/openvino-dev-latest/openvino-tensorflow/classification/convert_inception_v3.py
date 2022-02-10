import tensorflow as tf
import tensorflow_hub as hub

import os
label_file_path = os.getcwd() + "/../data/ImageNetLabels.txt"

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/4")
])
m.build([None, 299, 299, 3])  # Batch input shape.

m.save("../data/inception_v3")

tf.keras.utils.get_file(
        label_file_path,
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')