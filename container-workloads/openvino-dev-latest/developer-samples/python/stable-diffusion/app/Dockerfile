# Dockerfile
FROM python:3.8-slim-buster

WORKDIR /app

ENV https_proxy=http://proxy-dmz.intel.com:912
ENV http_proxy=http://proxy-dmz.intel.com:911
RUN apt-get update && apt-get install wget libgl1-mesa-glx ffmpeg libsm6 libxext6 -y

RUN python -m pip install --upgrade pip
RUN pip install openvino-dev[onnx,pytorch]==2023.0.0

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade watchdog
COPY . .

RUN mkdir models

WORKDIR /app/models

RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/text_encoder.bin
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/text_encoder.xml
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/unet.xml
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/unet.bin
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/vae_encoder.xml
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/vae_encoder.bin
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/vae_decoder.xml
RUN wget https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/resolve/main/vae_decoder.bin

WORKDIR /app

ENV https_proxy=
ENV http_proxy=

EXPOSE 5000 8000

CMD ["python", "app.py"]
