"""
 Copyright (C) 2018-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
# pylint: disable=E1101

import json
import os
import logging as log
from argparse import ArgumentParser
import sys
import time
import numpy as np

# OpenVINO 2.0 Inference API Pipeline Upgrade
from openvino.runtime import AsyncInferQueue, Core, CompiledModel
import cv2

log.basicConfig(
    format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
)


def build_argparser():
    """Input Arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Path to an .xml file with a trained model.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-i", "--input", help="Path to input video file.", required=True, type=str
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target infer device to; CPU, GPU, MYRIAD, or HDDL."
        "(CPU by default).",
        default="CPU",
        type=str,
    )
    parser.add_argument(
        "-l", "--labels", help="Labels mapping file.", default=None, type=str
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        help="Probability threshold for detection filtering.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Location to store the results of the processing",
        default=None,
        required=True,
        type=str,
    )

    parser.add_argument(
        "-nireq",
        "--number_infer_requests",
        help="Number of parallel inference requests (default is 2).",
        type=int,
        required=False,
        default=2,
    )
    return parser


def process_boxes(
    frame_count,
    res,
    labels_map,
    prob_threshold,
    initial_w,
    initial_h,
    result_file,
    infer_time,
):
    """Extract results - bounding boxes, labels and save to file."""
    for obj in res:
        dims = ""
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            class_id = int(obj[1])
            det_label = labels_map[class_id] if labels_map else "class=" + str(class_id)
            dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {det_label} {est} {time} \n".format(
                frame_id=frame_count,
                xmin=int(obj[3] * initial_w),
                ymin=int(obj[4] * initial_h),
                xmax=int(obj[5] * initial_w),
                ymax=int(obj[6] * initial_h),
                class_id=class_id,
                det_label=det_label,
                est=round(obj[2] * 100, 1),
                time=infer_time,
            )
            result_file.write(dims)


def preprocess_video(
    video_input, pre_infer_file, processed_vid, batch_size, channels, height, width
):
    """Transform video to resize, transpose, reshape to model inputs"""
    cap = cv2.VideoCapture(video_input)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    chunk_size = batch_size * channels * width * height
    id_ = 0
    log.info("Preprocessing %d frames and saving to file.", video_len)
    with open(processed_vid, "w+b") as file:
        time_start = time.time()
        while cap.isOpened():
            ret, next_frame = cap.read()
            if not ret:
                break
            in_frame = cv2.resize(next_frame, (width, height))
            in_frame = in_frame.transpose(
                (2, 0, 1)
            )  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((batch_size, channels, height, width))
            bin_frame = bytearray(in_frame)
            file.write(bin_frame)
            id_ += 1
            #if id_ % 10 == 0:
            #    progressUpdate(pre_infer_file, time.time() - time_start, id_, video_len)
    cap.release()
    log.info("Completed Preprocessing.")
    return chunk_size, video_width, video_height, video_len


def callback(request, callback_args):
    """Callback for each infer request in the queue"""
    frame_id, video_w, video_h, threshold, result_file, labels_map = callback_args
    output_tensor = request.get_output_tensor()
    process_boxes(
        frame_id,
        output_tensor.data[0][0],
        labels_map,
        threshold,
        video_w,
        video_h,
        result_file,
        round(request.latency, 2),
    )


def main():
    """EntryPoint for the program."""
    args = build_argparser().parse_args()

    # Create OpenVINO™ Runtime Core
    core = Core()

    # Compile the model optimized for Thorugput, can be toggled for Latency bu changing config value
    compiled_model = core.compile_model(
        model=args.model,
        device_name=args.device,
        config={"PERFORMANCE_HINT": "THROUGHPUT"},
    )
    if isinstance(compiled_model, CompiledModel):
        log.info(
            "Successfully Compiled model (%s) for (%s) device", args.model, args.device
        )

    # create async queue with optimal number of infer requests
    infer_queue = AsyncInferQueue(compiled_model)
    infer_queue.set_callback(callback)

    # Get input nodes.
    input_layer = compiled_model.input(0)

    # Setup output file for the program
    result_file = open(
        os.path.join(args.output_dir, "output.txt"), "w", encoding="utf-8"
    )
    pre_infer_file = os.path.join(args.output_dir, "pre_progress.txt")
    infer_file = os.path.join(args.output_dir, "i_progress.txt")
    processed_vid = os.path.join(args.output_dir, "processed_vid.bin")

    # Input layer: batch size(n), channels (c), height(h), width(w)
    batch_size, channels, height, width = input_layer.shape
    log.info(
        "Model Input Info - batch size:%d channels:%d height:%d width:%d",
        batch_size,
        channels,
        height,
        width,
    )

    # Preprocess video file
    chunk_size, video_width, video_height, video_len = preprocess_video(
        args.input, pre_infer_file, processed_vid, batch_size, channels, height, width
    )
    log.info(
        "Preprocess info - chunk_size:%d video_width:%d video_height:%d video_len:%d",
        chunk_size,
        video_width,
        video_height,
        video_len,
    )

    # Read labels file
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as file:
            labels_map = [x.strip() for x in file]
            log.info("Completed reading labels file: %s", args.labels)
    else:
        labels_map = None

    # Start Async Inference
    log.info("Starting Async Inference requests")
    infer_time_start = time.time()
    frame_count = 0
    with open(processed_vid, "rb") as data:
        while frame_count < video_len:
            byte = data.read(chunk_size)
            if not byte == b"":
                deserialized_bytes = np.frombuffer(byte, dtype=np.uint8)
                in_frame = np.reshape(
                    deserialized_bytes, newshape=(batch_size, channels, height, width)
                )
                callback_data = (
                    frame_count,
                    video_width,
                    video_height,
                    args.prob_threshold,
                    result_file,
                    labels_map,
                )
                infer_queue.start_async(
                    inputs={input_layer.any_name: in_frame}, userdata=callback_data
                )
                frame_count += 1


    infer_queue.wait_all()
    log.info("Completed all async requests")

    # Write out stats
    total_time = round(time.time() - infer_time_start, 2)
    stats = {}
    stats["frames"] = str(frame_count)
    with open(
        os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(stats, file)

    result_file.close()

    with open(os.path.join(args.output_dir, f'performance.txt'), 'w') as f:

            f.write('Throughput: {:.3g} FPS \n'.format(frame_count/total_time))
            f.write('Latency: {:.3f} ms\n'.format(total_time*1000))

   
    f.close()

    # clean-up
    del compiled_model
    os.remove(processed_vid)


if __name__ == "__main__":
    sys.exit(main() or 0)
