"""
 Copyright (C) 2018-2021 Intel Corporation

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

import os
import cv2
import numpy as np
from glob import glob

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger

def set_inputs(paths_to_input, batch_size, app_input_info, requests):
  requests_input_data = get_inputs(paths_to_input, batch_size, app_input_info, requests)
  for i in range(len(requests)):
    inputs = requests[i].input_blobs
    for k, v in requests_input_data[i].items():
        if k not in inputs.keys():
            raise Exception("No input with name {} found!".format(k))
        inputs[k].buffer[:] = v

def get_inputs(paths_to_input, batch_size, app_input_info, requests):
    input_image_sizes = {}
    for key in sorted(app_input_info.keys()):
        info = app_input_info[key]
        if info.is_image:
            input_image_sizes[key] = (info.width, info.height)
        logger.info("Network input '{}' precision {}, dimensions ({}): {}".format(key,
                                                                                  info.precision,
                                                                                  info.layout,
                                                                                  " ".join(str(x) for x in
                                                                                           info.shape)))

    images_count = len(input_image_sizes.keys())
    binaries_count = len(app_input_info) - images_count

    image_files = list()
    binary_files = list()

    if paths_to_input:
        image_files = get_files_by_extensions(paths_to_input, IMAGE_EXTENSIONS)
        image_files.sort()
        binary_files = get_files_by_extensions(paths_to_input, BINARY_EXTENSIONS)
        binary_files.sort()

    if (len(image_files) == 0) and (len(binary_files) == 0):
        logger.warning("No input files were given: all inputs will be filled with random values!")
    else:
        binary_to_be_used = binaries_count * batch_size * len(requests)
        if binary_to_be_used > 0 and len(binary_files) == 0:
            logger.warning("No supported binary inputs found! Please check your file extensions: {}".format(
                ",".join(BINARY_EXTENSIONS)))
        elif binary_to_be_used > len(binary_files):
            logger.warning(
                "Some binary input files will be duplicated: {} files are required, but only {} were provided".format(
                    binary_to_be_used, len(binary_files)))
        elif binary_to_be_used < len(binary_files):
            logger.warning(
                "Some binary input files will be ignored: only {} files are required from {}".format(binary_to_be_used,
                                                                                                     len(binary_files)))

        images_to_be_used = images_count * batch_size * len(requests)
        if images_to_be_used > 0 and len(image_files) == 0:
            logger.warning("No supported image inputs found! Please check your file extensions: {}".format(
                ",".join(IMAGE_EXTENSIONS)))
        elif images_to_be_used > len(image_files):
            logger.warning(
                "Some image input files will be duplicated: {} files are required, but only {} were provided".format(
                    images_to_be_used, len(image_files)))
        elif images_to_be_used < len(image_files):
            logger.warning(
                "Some image input files will be ignored: only {} files are required from {}".format(images_to_be_used,
                                                                                                    len(image_files)))

    requests_input_data = []
    for request_id in range(0, len(requests)):
        logger.info("Infer Request {} filling".format(request_id))
        input_data = {}
        keys = list(sorted(app_input_info.keys()))
        for key in keys:
            info = app_input_info[key]
            if info.is_image:
                # input is image
                if len(image_files) > 0:
                    input_data[key] = fill_blob_with_image(image_files, request_id, batch_size, keys.index(key),
                                                           len(keys), info)
                    continue

            # input is binary
            if len(binary_files):
                input_data[key] = fill_blob_with_binary(binary_files, request_id, batch_size, keys.index(key),
                                                        len(keys), info)
                continue

            # most likely input is image info
            if info.is_image_info and len(input_image_sizes) == 1:
                image_size = input_image_sizes[list(input_image_sizes.keys()).pop()]
                logger.info("Fill input '" + key + "' with image size " + str(image_size[0]) + "x" +
                            str(image_size[1]))
                input_data[key] = fill_blob_with_image_info(image_size, info)
                continue

            # fill with random data
            logger.info("Fill input '{}' with random values ({} is expected)".format(key, "image"
                if info.is_image else "some binary data"))
            input_data[key] = fill_blob_with_random(info)

        requests_input_data.append(input_data)

    return requests_input_data


def get_files_by_extensions(paths_to_input, extensions):
    get_extension = lambda file_path: file_path.split(".")[-1].upper()

    input_files = list()
    for path_to_input in paths_to_input:
        if os.path.isfile(path_to_input):
            files = [os.path.normpath(path_to_input)]
        else:
            path = os.path.join(path_to_input, '*')
            files = glob(path, recursive=True)
        for file in files:
            file_extension = get_extension(file)
            if file_extension in extensions:
                input_files.append(file)

    return input_files

def fill_blob_with_image(image_paths, request_id, batch_size, input_id, input_size, info):
    shape = info.shape
    images = np.ndarray(shape)
    image_index = request_id * batch_size * input_size + input_id
    for b in range(batch_size):
        image_index %= len(image_paths)
        image_filename = image_paths[image_index]
        logger.info('Prepare image {}'.format(image_filename))
        image = cv2.imread(image_filename)
        new_im_size = tuple((info.width, info.height))
        if image.shape[:-1] != new_im_size:
            logger.warning("Image is resized from ({}) to ({})".format(image.shape[:-1], new_im_size))
            image = cv2.resize(image, new_im_size)
        if info.layout in ['NCHW', 'CHW']:
            image = image.transpose((2, 0, 1))
        images[b] = image

        image_index += input_size
    return images

def get_dtype(precision):
    format_map = {
      'FP32' : np.float32,
      'I32'  : np.int32,
      'I64'  : np.int64,
      'FP16' : np.float16,
      'I16'  : np.int16,
      'U16'  : np.uint16,
      'I8'   : np.int8,
      'U8'   : np.uint8,
    }
    if precision in format_map.keys():
        return format_map[precision]
    raise Exception("Can't find data type for precision: " + precision)

def fill_blob_with_binary(binary_paths, request_id, batch_size, input_id, input_size, info):
    binaries = np.ndarray(info.shape)
    shape = info.shape.copy()
    if 'N' in info.layout:
        shape[info.layout.index('N')] = 1
    binary_index = request_id * batch_size * input_size + input_id
    dtype = get_dtype(info.precision)
    for b in range(batch_size):
        binary_index %= len(binary_paths)
        binary_filename = binary_paths[binary_index]
        logger.info("Prepare binary file " + binary_filename)

        binary_file_size = os.path.getsize(binary_filename)
        blob_size = dtype().nbytes * int(np.prod(shape))
        if blob_size != binary_file_size:
            raise Exception(
                "File {} contains {} bytes but network expects {}".format(binary_filename, binary_file_size, blob_size))
        binaries[b] = np.reshape(np.fromfile(binary_filename, dtype), shape)
        binary_index += input_size

    return binaries


def fill_blob_with_image_info(image_size, layer):
    shape = layer.shape
    im_info = np.ndarray(shape)
    for b in range(shape[0]):
        for i in range(shape[1]):
            im_info[b][i] = image_size[i] if i in [0, 1] else 1

    return im_info

def fill_blob_with_random(layer):
    if layer.shape:
        return np.random.rand(*layer.shape).astype(get_dtype(layer.precision))
    return (get_dtype(layer.precision))(np.random.rand())
