import cv2
import sys
import os
import time
import io
from argparse import ArgumentParser
from ast import literal_eval as make_tuple
from pathlib import Path
from qarpo.demoutils import *

# Multiplication factor to compute time interval for uploading snapshots to the cloud
MULTIPLICATION_FACTOR = 5

# Azure Blob container name
CONTAINER_NAME = 'store-aisle-monitor-snapshots'

# To get current working directory
CWD = os.getcwd()

# Creates subdirectory to save output snapshots
Path(CWD + '/output_snapshots/').mkdir(parents=True, exist_ok=True)

def apply_time_stamp_and_save(image, people_count, upload_azure):
    """
    Saves snapshots with timestamps.
    """
    current_date_time = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = current_date_time + "_PCount_" + str(people_count) + ".png"
    file_path = CWD + "/output_snapshots/"
    local_file_name = "output_" + file_name
    file_name = file_path + local_file_name
    cv2.imwrite(file_name, image)
    if upload_azure is 1:
        upload_snapshot(file_path, local_file_name)


def create_cloud_container(account_name, account_key):
    """
    Creates a BlockBlobService container on cloud.
    """
    global BLOCK_BLOB_SERVICE

    # Create the BlockBlobService to call the Blob service for the storage account
    BLOCK_BLOB_SERVICE = BlockBlobService(account_name, account_key)
    # Create BlockBlobService container
    BLOCK_BLOB_SERVICE.create_container(CONTAINER_NAME)
    # Set the permission so that the blobs are public
    BLOCK_BLOB_SERVICE.set_container_acl(CONTAINER_NAME, public_access=PublicAccess.Container)


def upload_snapshot(file_path, local_file_name):
    """
    Uploads snapshots to cloud storage container.
    """
    try:

        full_path_to_file = file_path + local_file_name
        print("\nUploading to cloud storage as blob : " + local_file_name)
        # Upload the snapshot, with local_file_name as the blob name
        BLOCK_BLOB_SERVICE.create_blob_from_path(CONTAINER_NAME, local_file_name, full_path_to_file)

    except Exception as e:
        print(e)


def post_process( input_stream, input_file, progress_file, output_file):
    # Read the input file and store in data dict
    data = {}
    with open(input_file, 'r') as file:
        for line in file.readlines():
            data_list = line.split(";")
            frame = int(data_list[0])
            data[frame] = {}
            data[frame]["People_Count"] = data_list[-1]
            data[frame]["Det_Time"] = data_list[-2]
            l = []
            for index in range(int(data_list[-1])):
                l.append(data_list[1 + (index * 3):4 + (index * 3)])
            data[frame]["rect"] = l

    del (frame)

    # Read the Video File and append the rect
    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    store_aisle = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'),
                                  fps, (initial_w, initial_h), True)

    frame_count = 1
    start_time = time.time()
    ret, frame = cap.read()

    while ret:
        people_count = 0
        det_time = 0.0001
        if frame_count in data.keys():
            people_count = data[frame_count]["People_Count"][0]
            for rect in data[frame_count]['rect']:
                cv2.rectangle(frame, make_tuple(rect[0]), make_tuple(rect[1]), make_tuple(rect[2]), 2)
            det_time = float(data[frame_count]["Det_Time"])
        inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
        people_count_message = "People Count : " + str(people_count)
        cv2.putText(frame, inf_time_message, (15, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 1)
        cv2.putText(frame, people_count_message, (15, 65), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 1)
        store_aisle.write(frame)
        time_interval = MULTIPLICATION_FACTOR * fps
        if frame_count % time_interval == 0:
            apply_time_stamp_and_save(frame, people_count, 0)
        if frame_count >= video_len:
            break
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % 10 == 0:
            progressUpdate(progress_file, int(time.time()-start_time), frame_count, video_len)
     
        
def main():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type = str, required=True,
                        help = "Path to output directory")
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. Use 'cam' for "
                             "capturing video stream from camera",
                        required=True, type=str)

    args = parser.parse_args()
    
    job_id = os.environ['PBS_JOBID']
    input_file = args.input
    input_data = f"{args.output_dir}/output_{job_id}.txt"
    progress_data = f"{args.output_dir}/post_progress_{job_id}.txt"
    output_file = os.path.join(args.output_dir, f"store_aisle_{job_id}.mp4")
    
    print(f"input_data={input_data}")
    print(f"progress_data={progress_data}")

    post_process( input_file, input_data, progress_data, output_file)
    
if __name__ == '__main__':
    sys.exit(main() or 0)