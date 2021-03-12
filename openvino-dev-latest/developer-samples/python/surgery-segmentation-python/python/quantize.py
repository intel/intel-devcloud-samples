import os
import numpy as np
import cv2 as cv

from addict import Dict
from compression.graph import load_model, save_model
from compression.api.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.pipeline.initializer import create_pipeline

def video_to_frames(video_path):
    vidcap = cv.VideoCapture(video_path)
    frames = []
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else: 
            break 
    return frames

class DatasetsDataLoader(DataLoader):
 
    def __init__(self, config):
        super().__init__(config)
        self.images = video_to_frames(str(config['data_source']))

    @property
    def size(self):
        return len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        image = self.images[item]

        # Resize the input image to match the expected value
        cropHeight, cropWidth = (1024,1280)
        imgHeight, imgWidth = image.shape[0], image.shape[1] 
        startH = (imgHeight - cropHeight) // 2
        startW = (imgWidth - cropWidth) // 2
        image = image[startH:(startH+cropHeight),startW:(startW+cropWidth),:]

        # Convert from BGR to RGB since model expects RGB input
        rgb_image = image[:,:,[2,1,0]]

        # Prepare input for model
        rgb_image = np.expand_dims(np.transpose(image/255.0, [2, 0, 1]), 0)

        return (item, None), rgb_image

work_directory=os.environ['WORK_DIR']

# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': 'surgical_tools_parts',
    'model': work_directory + '/models/ov/FP16/surgical_tools_parts.xml',
    'weights': work_directory + '/models/ov/FP16/surgical_tools_parts.bin',
})

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

dataset_config = Dict({
    'data_source': work_directory + '/data/short_source.mp4', # Path to input data for quantization
})

# Quantization algorithm settings
algorithms = [
    {
        'name': 'DefaultQuantization', # Optimization algorithm name
        'params': {
            'target_device': 'CPU',
            'preset': 'performance', # Preset [performance (default), accuracy] which controls the quantization mode 
                                     # (symmetric and asymmetric respectively)
            'stat_subset_size': 300  # Size of subset to calculate activations statistics that can be used
                                     # for quantization parameters calculation.
        }
    }
]

# Load the model.
model = load_model(model_config)

# Initialize the data loader.
data_loader = DatasetsDataLoader(dataset_config)

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, None)

# Create a pipeline of compression algorithms.
pipeline = create_pipeline(algorithms, engine)

# Execute the pipeline.
compressed_model = pipeline.run(model)

# Save the compressed model.
save_model(compressed_model, work_directory + '/models/ov/INT8')
