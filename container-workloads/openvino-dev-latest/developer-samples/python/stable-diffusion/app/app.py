# -- coding: utf-8 --`
from flask import Flask, request, render_template, send_file
import logging
import yaml
import io
from PIL import Image
import random
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np
from openvino.runtime import Core
from prometheus_client import start_http_server, Gauge
import psutil
import time
import threading


# Create Flask app
app = Flask(__name__)

# Initialize logger
logging.basicConfig(filename='logs/app.log', level=logging.DEBUG)

# Load configuration
with open('resources/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if config['seed'] is None:
    config['seed'] = random.randint(0, 2**30)
np.random.seed(config['seed'])
if config['init_image'] is None:
    scheduler = LMSDiscreteScheduler(
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        beta_schedule=config['beta_schedule'],
        tensor_format="np"
    )
else:
    scheduler = PNDMScheduler(
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        beta_schedule=config['beta_schedule'],
        skip_prk_steps = True,
        tensor_format="np"
    )
engine = StableDiffusionEngine(
    model=config['model'],
    scheduler=scheduler,
    tokenizer=config['tokenizer'],
    device=config['device']
)

# Initialize Prometheus metrics
INFERENCE_TIME = Gauge('inference_time_seconds', 'Time spent processing text')
CPU_USAGE = Gauge('cpu_usage_percent', 'Current Usage of the CPU')
RAM_USAGE = Gauge('ram_usage_gb', 'Current RAM usage')

# Function to report system metrics
def report_metrics():
    while True:
        # Write CPU temperature and RAM usage to Prometheus metrics
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().used / (1024 ** 3))  # in GB
         # Wait for 5 seconds
        time.sleep(5)
# Create and start the metrics reporting thread
metrics_thread = threading.Thread(target=report_metrics, daemon=True)
metrics_thread.start()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        text_input = request.json['text_input']
        logging.info(f'Received text input: {text_input}')

        start_time = time.time()

        image = engine(
            prompt=text_input,
            init_image=None if config['init_image'] is None else cv2.imread(config['init_image']),
            mask=None if config['mask'] is None else cv2.imread(config['mask'], 0),
            strength=config['strength'],
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            eta=config['eta']
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f'Inference time: {inference_time} seconds')

        # Update Prometheus metrics
        INFERENCE_TIME.set(inference_time)

        cv2.imwrite(config['output'], image)

        # Convert the image to a PIL Image object and then to a BytesIO object
        #image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        with open(config['output'], 'rb') as f:
            img = f.read()

        # Convert the image to PNG format
        img_io = io.BytesIO(img)

        logging.info('Image generated successfully')

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logging.error(str(e))
        return {"error": str(e)}, 500

if __name__ == '__main__':
    # Start the Prometheus client server to expose metrics
    start_http_server(8000)

    app.run(debug=True, host='0.0.0.0', use_reloader=False)
