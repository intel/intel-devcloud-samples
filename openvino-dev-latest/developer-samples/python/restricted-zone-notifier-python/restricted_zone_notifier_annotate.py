import sys
import os
import cv2
import time
import numpy as np
import io
from argparse import ArgumentParser

from qarpo.demoutils import progressUpdate

class ResultData:
    frame_id=""
    xmin=""
    ymin=""
    xmax=""
    ymax=""
    det_label=""
    render_time=""
    det_time=""

def placeBoxes(frame, obj):
    color = (255, 255, 255)
    color1 = (0, 0, 255)
    warning = "HUMAN IN ASSEMBLY AREA: PAUSE THE MACHINE!"
    inf_time_message = "Inference time:" + obj.det_time + " ms"
    render_time_message = "OpenCV rendering time:" + obj.render_time + "ms"
    det_label_message = "Worker Safe :" + obj.det_label
    if (obj.det_label == "False"):
        cv2.putText(frame, warning, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, color1, 2)

    cv2.rectangle(frame, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), color1, 2)
    cv2.putText(frame, det_label_message, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    cv2.putText(frame, inf_time_message, (20, 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame

def post_process(input_stream, input_data, out_path, progress_data, scale_frame_rate, scale_resolution):
    post_process_time_start = time.time()
    cap = cv2.VideoCapture(input_stream)
    if cap.isOpened():   
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = int(scale_resolution*width)
        out_h = int(scale_resolution*height)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), 50.0 / scale_frame_rate, (out_w, out_h), True)
        #vw = cv2.VideoWriter(out_path, 0x00000021, 50.0 / scale_frame_rate, (out_w, out_h), True)
        #vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), 50.0 / scale_frame_rate, (int(scale_resolution*width), int(scale_resolution*height)), True)
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
    	print('failed to open input video stram')
    	return
    	
    f_input_data = open(input_data, "r")
    
    frame_count = 0
    while cap.isOpened():
    	# input data is saved for every frame
        ret, frame = cap.read()
        line = f_input_data.readline()
        
        if not ret or not line:
            break
        
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        
        # read input data for frame into object
        rd = ResultData()
        rd.frame_id, rd.xmin, rd.ymin, rd.xmax, rd.ymax, rd.det_label, rd.render_time, rd.det_time = line.split()

        if (frame_count % scale_frame_rate == 0):
            frame = placeBoxes(frame, rd)
            frame = cv2.resize(frame, (out_w, out_h))
            vw.write(frame)
            
        frame_count += 1
        
        # report progress
        progressUpdate(progress_data, int(time.time()-post_process_time_start), frame_count, video_len)

        key = cv2.waitKey(1)
        if key == 27:
            break
    print("Post processing time: {0} sec" .format(time.time()-post_process_time_start))
    cap.release()
    vw.release()

def main():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image. ")
    parser.add_argument("-o", "--output_dir", type = str, required=True,
                        help = "Path to output directory")
    parser.add_argument("-f", "--scale_frame_rate", type=float, default=1.0,
                        help="Output frame rate scale value (FPS=50.0/<val>)")
    parser.add_argument("-s", "--scale_resolution", type=float, default=1.0,
                        help="Output resolution scale value to (input_w*<val>,input_h*<val>)")

    args = parser.parse_args()
    
    job_id = os.environ['PBS_JOBID']
    input_data = f"{args.output_dir}/output_{job_id}.txt"
    progress_data = f"{args.output_dir}/v_progress_{job_id}.txt"
    #output_stream = f"{args.output_dir}/output_{job_id}.mp4"
    output_stream = f"{args.output_dir}/output_{job_id}.mp4"
    
    print(f"input_data={input_data}")
    print(f"progress_data={progress_data}")
    print(f"output_stream={output_stream}")
    print(f"args.scale_frame_rate={args.scale_frame_rate}")
    print(f"args.scale_resolution={args.scale_resolution}")

    post_process( args.input, input_data, output_stream, progress_data, args.scale_frame_rate, args.scale_resolution )

if __name__ == '__main__':
    main()
