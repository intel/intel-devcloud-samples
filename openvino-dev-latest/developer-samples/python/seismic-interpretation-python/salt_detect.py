from __future__ import print_function
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import shutil
import subprocess
from tqdm import tqdm
import numpy as np
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IECore
from pathlib import Path
from qarpo.demoutils import progressUpdate
import applicationMetricWriter


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Path to a directory with .xml and .bin file of the trained model.',
                        required=True,
                        type=str)
    
    parser.add_argument('-o', '--output_dir',
                        help='Location to store the results of the processing',
                        default=None,
                        required=True,
                        type=str)
    
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on; CPU, GPU, MYRIAD, or HDDL is acceptable.'
                             'Demo will look for a suitable plugin for specified device (CPU by default).',
                        default='CPU',
                        type=str)
    return parser


def main():
    job_id = str(os.environ['PBS_JOBID'])
    ir_name = "textureNet"
    ir_data_type = "FP32"
    args = build_argparser().parse_args()
    ir_out_dir = args.model
    print(ir_out_dir)
    filename = "./data/f3-sample.npy"
    dataset_name = 'F3'
    subsampl = 16 
    im_size = 65

    data = np.load(filename)
    data = np.moveaxis(data, -1, 0)
    data = np.ascontiguousarray(data,'float32')

    data_info = {}
    data_info['shape'] = data.shape

    def ls(N):  return np.linspace(0, N - 1, N, dtype='int')
    N0, N1, N2 = data.shape
    x0_range = ls(N0)
    x1_range = ls(N1)
    x2_range = ls(N2)
    pred_points = (x0_range[::subsampl], x1_range[::subsampl], x2_range[::subsampl])
    class_cube = data[::subsampl, ::subsampl, ::subsampl] * 0

    print(f"Loaded Data info: \n{data_info}")
    print(f"Class cube shape: {class_cube.shape}")

    n0,n1,n2 = class_cube.shape
    x0_grid, x1_grid, x2_grid = np.meshgrid(ls(n0,), ls(n1), ls(n2), indexing='ij')
    X0_grid, X1_grid, X2_grid = np.meshgrid(x0_range, x1_range, x2_range, indexing='ij')

    X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
    X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
    X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

    w = im_size//2

    from openvino.inference_engine import IECore

    model_xml = f'{ir_out_dir}/{ir_name}.xml'
    model_bin = f'{ir_out_dir}/{ir_name}.bin'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
    del net

    input_layer = next(iter(exec_net.input_info))
    output_layer = next(iter(exec_net.outputs))
    
    infer_time_start = time.time()
    for i in tqdm(range(X0_grid_sub.size)):
        
        x0, x1, x2 = x0_grid.ravel()[i], x1_grid.ravel()[i], x2_grid.ravel()[i]
        X0, X1, X2 = X0_grid_sub.ravel()[i], X1_grid_sub.ravel()[i], X2_grid_sub.ravel()[i]
        if X0>w and X1>w and X2>w and X0<N0-w+1 and X1<N1-w+1 and X2<N2-w+1:
            mini_cube = data[X0-w:X0+w+ 1, X1-w:X1+w+ 1, X2-w:X2+w+ 1]
            
            inf_time = time.time()
            
            out = exec_net.infer({input_layer: mini_cube})[output_layer]
            
            det_time = time.time() - inf_time
            applicationMetricWriter.send_inference_time(det_time*1000)
            
            out = out[:,:, out.shape[2]//2, out.shape[3]//2, out.shape[4]//2]
            out = np.squeeze(out)

            # Make one output pr output channel
            if type(class_cube) != type(list()):
                class_cube = np.split( np.repeat(class_cube[:,:,:,np.newaxis],out.size,3),out.size, axis=3)

            # Insert into output
            if out.size == 1:
                class_cube[0][x0, x1, x2] = out
            else:
                for i in range(out.size):
                    class_cube[i][x0,x1,x2] = out[i]
            
    total_time = time.time() - infer_time_start
    frame_count = 1.0/total_time
    with open(os.path.join(args.output_dir, f'stats_{job_id}.txt'), 'w') as f:
            f.write('{:.3g} \n'.format(total_time))
            f.write('{} \n'.format(frame_count))


    #Interpolation
    from scipy.interpolate import interpn
    N = X0_grid.size
    grid_output_cube = np.concatenate( [X0_grid.reshape([N, 1]), X1_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1)

    for i in tqdm(range(len(class_cube))):
        is_int = np.sum(np.unique(class_cube[i]).astype('float') - np.unique(class_cube[i]).astype('int32').astype('float') ) == 0
        class_cube[i] = interpn(pred_points, class_cube[i].astype('float').squeeze(), grid_output_cube, method='linear', fill_value=0, bounds_error=False)
        class_cube[i] = class_cube[i].reshape([x0_range.size, x1_range.size, x2_range.size])

        if is_int:
            class_cube[i] = class_cube[i].astype('int32')


    #Squeeze outputs
    for i in range(len(class_cube)):
        class_cube[i]= class_cube[i].squeeze()
        


    k1,k2,k3 = 40,200,50
    gx1 = data[k1,:,:]
    gy1 = class_cube[0][k1,:,:]
    gx2 = data[:,k2,:]
    gy2 = class_cube[0][:,k2,:]
    gx3 = data[:,:,k3]
    gy3 = class_cube[0][:,:,k3]

    rows = 3
    cols = 2
    axes=[]
    fig = plt.figure(figsize=(12,12))

    axes.append( fig.add_subplot(rows, cols, 1) )
    subplot_title=("Inline")
    axes[0].set_title(subplot_title)  
    plt.imshow(gx3,cmap=plt.cm.gray)

    axes.append( fig.add_subplot(rows, cols, 2) )
    subplot_title=("Prediction")
    axes[1].set_title(subplot_title)  
    plt.imshow(gy3,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)

    axes.append( fig.add_subplot(rows, cols, 3) )
    subplot_title=("Cross Slice")
    axes[2].set_title(subplot_title)  
    plt.imshow(gx2,aspect=1.5,cmap=plt.cm.gray)

    axes.append( fig.add_subplot(rows, cols, 4) )
    subplot_title=("Prediction")
    axes[3].set_title(subplot_title)  
    plt.imshow(gy2,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)

    axes.append( fig.add_subplot(rows, cols, 5) )
    subplot_title=("Time Slice")
    axes[4].set_title(subplot_title)  
    plt.imshow(gx1,aspect=1.5,cmap=plt.cm.gray)

    axes.append( fig.add_subplot(rows, cols, 6) )
    subplot_title=("Prediction")
    axes[5].set_title(subplot_title)  
    plt.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)

    plot_path = args.output_dir+'/combined-plot.png'
    fig.tight_layout()    
    plt.savefig(plot_path)
    applicationMetricWriter.send_application_metrics(model_xml, args.device)
        
if __name__ == '__main__':
    sys.exit(main() or 0)
