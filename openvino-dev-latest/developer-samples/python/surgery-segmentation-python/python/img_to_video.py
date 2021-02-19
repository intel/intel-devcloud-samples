import cv2
import argparse
import os
import numpy as np

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False,
                default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False,
                default='source.mp4', help="output video file")
ap.add_argument("-p", "--path", required=False,
                default='.', help="path to images")
args = vars(ap.parse_args())

# Arguments
dir_path = args['path']
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
#cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

check = []
count = 0
video_length = 25

for image in images:
    count += 1

    img_path = os.path.join(dir_path, image)

    img = cv2.imread(img_path)


    # Write out frame to video
    out.write(img)

    #cv2.imshow('video', img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

    if count == video_length:
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))