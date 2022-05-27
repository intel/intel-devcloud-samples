import numpy as np
import onnxruntime as rt
import cv2
import time
import argparse
import os


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


def preprocessing(frame):
    resized_frame = cv2.resize(frame, (416, 416))
    preproc_frame = np.asarray(resized_frame)
    preproc_frame = preproc_frame.astype(np.float32)
    preproc_frame = preproc_frame.transpose(2, 0, 1)
    preproc_frame = preproc_frame.reshape(1, 3, 416, 416)
    return preproc_frame


def postprocessing(predictions, x_scale, y_scale, frame):
    clut = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 200, 0)]
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    label = ["unprotected", "bunny suit", "glasses", "head", "robot"]
    numClasses = len(label)
    existingLabels = {l: [] for l in label}

    for cy in range(0, 13):
        for cx in range(0, 13):
            for b in range(0, 5):
                channel = b * (numClasses + 5)
                tx = predictions[channel][cy][cx]
                ty = predictions[channel + 1][cy][cx]
                tw = predictions[channel + 2][cy][cx]
                th = predictions[channel + 3][cy][cx]
                tc = predictions[channel + 4][cy][cx]

                x = (float(cx) + sigmoid(tx)) * 32
                y = (float(cy) + sigmoid(ty)) * 32

                w = np.exp(tw) * 32 * anchors[2 * b]
                h = np.exp(th) * 32 * anchors[2 * b + 1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0, numClasses):
                    classes[c] = predictions[channel + 5 + c][cy][cx]
                classes = softmax(classes)
                detectedClass = classes.argmax()

                if 0.45 < classes[detectedClass] * confidence:
                    color = clut[detectedClass]
                    x = (x - w / 2) * x_scale
                    y = (y - h / 2) * y_scale
                    w *= x_scale
                    h *= y_scale

                    labelX = int((x + x + w) / 2)
                    labelY = int((y + y + h) / 2)
                    addLabel = True
                    labThreshold = 40
                    for point in existingLabels[label[detectedClass]]:
                        if (
                            labelX < point[0] + labThreshold
                            and labelX > point[0] - labThreshold
                            and labelY < point[1] + labThreshold
                            and labelY > point[1] - labThreshold
                        ):
                            addLabel = False
                    if addLabel:
                        cv2.rectangle(
                            frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2
                        )
                        cv2.rectangle(
                            frame,
                            (int(x), int(y - 13)),
                            (int(x) + 9 * len(label[detectedClass]), int(y)),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label[detectedClass],
                            (int(x) + 2, int(y) - 3),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                        existingLabels[label[detectedClass]].append((labelX, labelY))
                    print(
                        "{} detected in frame {}".format(
                            label[detectedClass], frameCount
                        )
                    )

    return frame


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Path to input video", required=True)
parser.add_argument("-m", help="Path to ONNX model", required=True)
parser.add_argument(
    "-d",
    default="CPU",
    help="Device(CPU_FP32,GPU_FP32,GPU_FP16,VAD-M_FP16,VAD-F_FP32,MYRIAD_FP16 or CPU for default EP",
)

args = parser.parse_args()

if not os.path.exists(args.i):
    print("Input video doesn't exists! Please supply proper path of video to -i.")
else:
    input_video = args.i

if not os.path.exists(args.m):
    print("input model doesn't exist! Please supply proper path of model to -m")
else:
    model = args.m
# need to have device list and if device isn't found then put error
device = args.d

output_path = "/mount_folder"
print("This is output path: ", output_path)

if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError:
        print(" Failed to Create Directory %s" % output_path)
    else:
        print("Output directory %s was successfully created" % output_path)


output_file = "output_results_" + device + ".mp4"
output_path_file = output_path + "/" + output_file

if not os.path.exists(output_path_file):
    open(output_path_file, "w").close()


# used to track application's progress
inference_file = output_path + "inference_progress_" + ".txt"

"""ONNX Runtime - OpenVINO Calls for inference setup"""



# Disable ONNX optimizations and use OpenVINO's for best perf
graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL



# create inference session
print("Device: ", device)
result_file_name = "/mount_folder/" + "performance.txt"

if device == "CPU":
    sess = rt.InferenceSession(model, providers=["CPUExecutionProvider"])
    f = open(result_file_name, "w")
    f.write("Default Execution Provider for ONNXRT \n")
else:
    sess = rt.InferenceSession(
        model,
        providers=["OpenVINOExecutionProvider"],
        provider_options=[{"device_type": device}],
    )
    f = open(result_file_name, "w")
    f.write("Openvino Integration with ONNXRT \n")

f.write("Device: {} \n".format(device))
print(sess.get_providers())
# Get all inputs in the model
input_name = sess.get_inputs()[0].name
""" -------------------------------------------------"""
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_scale = float(width) / 416.0
y_scale = float(height) / 416.0
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"avc1")
output_video = cv2.VideoWriter(output_path_file, fourcc, float(17.0), (640, 360))

frameCount = 1
# first execution is the longest
frames = []
preproc_frames = []
frame = cap.read()
start = time.time()
while cap.isOpened() and frameCount < video_length - 1:
    l_start = time.time()
    _, _ = cap.read()
    frames = []
    preproc_frames = []
    for i in range(16):
        ret, frame = cap.read()
        # if even then capture the frame so that we're processing every other frame
        if i % 2 == 0:
            if frame is not None:
                frames.append(frame)
                preproc_frames.append(preprocessing(frame))
            else:
                continue

    # concatenate images into a batch
    # try:
    batch_arr = np.concatenate((preproc_frames), 0)
    # except:pass

    # run inference
    predictions = sess.run(None, {input_name: batch_arr.astype(np.float32)})
    predictions = np.asarray(predictions, dtype=np.float32)

    for i in range(8):
        try:
            postproc_frame = postprocessing(
                predictions[0][i], x_scale, y_scale, frames[i]
            )
            output_video.write(postproc_frame)
        except:
            pass

    frameCount += 17

total_time = time.time() - start

with open(os.path.join(output_path, "performance.txt"), "a") as f:
    f.write("Total Inference Engine Processing Time in Seconds: {:.3g} \n".format(total_time))
    f.write("FPS(Includes Preprocessing and Postprocessing): {} \n".format(frameCount//total_time))

output_video.release()
