import io
import os
import sys
import time
from argparse import ArgumentParser 
import cv2
import numpy as np
#from qarpo.demoutils import progressUpdate


class ResultData:
    frame_id = ""
    xmin = ""
    ymin = ""
    xmax = ""
    ymax = ""
    class_id = ""
    det_label = ""
    prob = ""
    det_time = ""


def placeBoxes(frame, rd):
    is_async_mode = True
    # Draw box and label\class_id
    inf_time_message = (
        r"Inference time: N\A for async mode"
        if is_async_mode
        else f"Inference time: {rd.det_time * 1000:.3f} ms"
    )
    async_mode_message = (
        f"Async mode is on. Processing request {rd.frame_id}"
        if is_async_mode
        else f"Async mode is off. Processing request {rd.frame_id}"
    )
    color = (
        min(int(rd.class_id) * 7, 255),
        min(int(rd.class_id) * 35, 255),
        min(int(rd.class_id) * 5, 255),
    )
    cv2.rectangle(
        frame, (int(rd.xmin), int(rd.ymin)), (int(rd.xmax), int(rd.ymax)), color, 3
    )
    cv2.putText(
        frame,
        rd.det_label + " " + str(round(float(rd.prob), 1)) + " %",
        (int(rd.xmin), int(rd.ymin) - 7),
        cv2.FONT_HERSHEY_COMPLEX,
        1.2,
        color,
        2,
    )
    cv2.putText(
        frame, inf_time_message, (20, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 10, 10), 2
    )
    cv2.putText(
        frame,
        async_mode_message,
        (20, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (10, 10, 200),
        2,
    )

    return frame


def post_process(
    input_stream,
    input_data,
    out_path,
    progress_data,
    scale_frame_rate,
    scale_resolution,
):
    post_process_time_start = time.time()
    cap = cv2.VideoCapture(input_stream)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = int(scale_resolution * width)
        out_h = int(scale_resolution * height)
        # vw = cv2.VideoWriter(out_path, 0x00000021, 50.0 / scale_frame_rate, (out_w, out_h), True)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), 50.0 / scale_frame_rate, (out_w, out_h), True)
        '''vw = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"vp80"),
            50.0 / scale_frame_rate,
            (out_w, out_h),
            True,
        )'''
        # vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 50.0 / scale_frame_rate, (out_w, out_h), True)
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        print("failed to open input video stream")
        return

    f_input_data = open(input_data)

    frame_count = 0
    input_frame_num = -1
    while cap.isOpened():
        # input data is saved for every frame
        ret, frame = cap.read()

        if not ret:
            break

        initial_w = cap.get(3)
        initial_h = cap.get(4)

        rd = None
        while input_frame_num < frame_count:
            line = f_input_data.readline()
            if not line:
                break
            # read input data for frame into object
            rd = ResultData()
            (
                rd.frame_id,
                rd.xmin,
                rd.ymin,
                rd.xmax,
                rd.ymax,
                rd.class_id,
                rd.det_label,
                rd.prob,
                rd.det_time,
            ) = line.split()
            frame = placeBoxes(frame, rd)
            input_frame_num = int(rd.frame_id)

        if frame_count % scale_frame_rate == 0:
            frame = cv2.resize(frame, (out_w, out_h))
            vw.write(frame)

        frame_count += 1

        # report progress
        '''progressUpdate(
            progress_data,
            int(time.time() - post_process_time_start),
            frame_count,
            video_len,
        )'''
       # print("OpenCV Version: ", cv2.__version__)
        #key = cv2.waitKey(1)
        #if key == 27:
           # break
    print(f"Post processing time: {time.time() - post_process_time_start} sec")
    cap.release()
    vw.release()


def main():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to video file or image. "
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-f",
        "--scale_frame_rate",
        type=float,
        default=1.0,
        help="Output frame rate scale value (FPS=50.0/<val>)",
    )
    parser.add_argument(
        "-s",
        "--scale_resolution",
        type=float,
        default=1.0,
        help="Output resolution scale value to (input_w*<val>,input_h*<val>)",
    )

    args = parser.parse_args()

    #job_id = str(os.environ["PBS_JOBID"]).split(".")[0]
    input_data = f"{args.output_dir}/output.txt"
    progress_data = f"{args.output_dir}/post_progress.txt"
    #output_stream = f"{args.output_dir}/output.webm"
    output_stream = f"{args.output_dir}/output.mp4"

    print(f"input_data={input_data}")
    print(f"progress_data={progress_data}")
    print(f"output_stream={output_stream}")
    print(f"args.scale_frame_rate={args.scale_frame_rate}")
    print(f"args.scale_resolution={args.scale_resolution}")

    post_process(
        args.input,
        input_data,
        output_stream,
        progress_data,
        args.scale_frame_rate,
        args.scale_resolution,
    )


if __name__ == "__main__":
    main()
