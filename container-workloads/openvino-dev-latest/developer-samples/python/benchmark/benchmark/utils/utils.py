"""
 Copyright (C) 2018-2020 Intel Corporation

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
from openvino.inference_engine import IENetwork,IECore

from .constants import DEVICE_DURATION_IN_SECS, UNKNOWN_DEVICE_TYPE, \
    CPU_DEVICE_NAME, GPU_DEVICE_NAME
from .inputs_filling import is_image
from .logging import logger

import json
import re

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(step_id=0)
def next_step(additional_info='', step_id=0):
    step_names = {
        1: "Parsing and validating input arguments",
        2: "Loading Inference Engine",
        3: "Setting device configuration",
        4: "Reading network files",
        5: "Resizing network to match image sizes and given batch",
        6: "Configuring input of the model",
        7: "Loading the model to the device",
        8: "Setting optimal runtime parameters",
        9: "Creating infer requests and filling input blobs with images",
        10: "Measuring performance",
        11: "Dumping statistics report",
    }
    if step_id != 0:
        next_step.step_id = step_id
    else:
        next_step.step_id += 1

    if next_step.step_id not in step_names.keys():
        raise Exception('Step ID {} is out of total steps number '.format(next_step.step_id, str(len(step_names))))

    step_info_template = '[Step {}/{}] {}'
    step_name = step_names[next_step.step_id] + (' ({})'.format(additional_info) if additional_info else '')
    step_info_template = step_info_template.format(next_step.step_id, len(step_names), step_name)
    print(step_info_template)


def config_network_inputs(ie_network: IENetwork):
    input_info = ie_network.input_info

    for key in input_info.keys():
        if is_image(input_info[key].input_data):
            # Set the precision of input data provided by the user
            # Should be called before load of the network to the plugin
            input_info[key].precision = 'U8'


def get_number_iterations(number_iterations: int, nireq: int, api_type: str):
    niter = number_iterations

    if api_type == 'async' and niter:
        niter = int((niter + nireq - 1) / nireq) * nireq
        if number_iterations != niter:
            logger.warning('Number of iterations was aligned by request number '
                        'from {} to {} using number of requests {}'.format(number_iterations, niter, nireq))

    return niter


def get_duration_seconds(time, number_iterations, device):
    if time:
        # time limit
        return time

    if not number_iterations:
        return get_duration_in_secs(device)
    return 0


def get_duration_in_milliseconds(duration):
    return duration * 1000


def get_duration_in_secs(target_device):
    duration = 0
    for device in DEVICE_DURATION_IN_SECS:
        if device in target_device:
            duration = max(duration, DEVICE_DURATION_IN_SECS[device])

    if duration == 0:
        duration = DEVICE_DURATION_IN_SECS[UNKNOWN_DEVICE_TYPE]
        logger.warn('Default duration {} seconds is used for unknown device {}'.format(duration, target_device))

    return duration


def parse_devices(device_string):
    if device_string in ['MULTI', 'HETERO']:
        return list()
    devices = device_string
    if ':' in devices:
        devices = devices.partition(':')[2]
    return [d[:d.index('(')] if '(' in d else
            d[:d.index('.')] if '.' in d else d for d in devices.split(',')]


def parse_nstreams_value_per_device(devices, values_string):
    # Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
        return result
    device_value_strings = values_string.split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            device_name = device_value_vec[0]
            nstreams = device_value_vec[1]
            if device_name in devices:
                result[device_name] = nstreams
            else:
                raise Exception("Can't set nstreams value " + str(nstreams) +
                                " for device '" + device_name + "'! Incorrect device name!");
        elif len(device_value_vec) == 1:
            nstreams = device_value_vec[0]
            for device in devices:
                result[device] = nstreams
        elif not device_value_vec:
            raise Exception('Unknown string format: ' + values_string)
    return result


def process_help_inference_string(benchmark_app):
    output_string = 'Start inference {}ronously'.format(benchmark_app.api_type)
    if benchmark_app.api_type == 'async':
        output_string += ', {} inference requests'.format(benchmark_app.nireq)

        device_ss = ''
        if CPU_DEVICE_NAME in benchmark_app.device:
            device_ss += str(benchmark_app.ie.get_config(CPU_DEVICE_NAME, 'CPU_THROUGHPUT_STREAMS'))
            device_ss += ' streams for {}'.format(CPU_DEVICE_NAME)
        if GPU_DEVICE_NAME in benchmark_app.device:
            device_ss += ', ' if device_ss else ''
            device_ss += str(benchmark_app.ie.get_config(GPU_DEVICE_NAME, 'GPU_THROUGHPUT_STREAMS'))
            device_ss += ' streams for {}'.format(GPU_DEVICE_NAME)

        if device_ss:
            output_string += ' using ' + device_ss

    limits = ''

    if benchmark_app.niter and not benchmark_app.duration_seconds:
        limits += '{} iterations'.format(benchmark_app.niter)

    if benchmark_app.duration_seconds:
        limits += '{} ms duration'.format(get_duration_in_milliseconds(benchmark_app.duration_seconds))
    if limits:
        output_string += ', limits: ' + limits

    return output_string


def dump_exec_graph(exe_network, exec_graph_path):
    try:
        exec_graph_info = exe_network.get_exec_graph_info()
        exec_graph_info.serialize(exec_graph_path)
        logger.info('Executable graph is stored to {}'.format(exec_graph_path))
        del exec_graph_info
    except Exception as e:
        logger.exception(e)


def print_perf_counters(perf_counts_list):
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = 0
        total_time_cpu = 0
        logger.info("Performance counts for {}-th infer request".format(ni))
        for layer, stats in sorted(perf_counts.items(), key=lambda x: x[1]['execution_index']):
            max_layer_name = 30
            print("{:<30}{:<15}{:<30}{:<20}{:<20}{:<20}".format(
                layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer,
                stats['status'],
                'layerType: ' + str(stats['layer_type']),
                'realTime: ' + str(stats['real_time']),
                'cpu: ' + str(stats['cpu_time']),
                'execType: ' + str(stats['exec_type'])))
            total_time += stats['real_time']
            total_time_cpu += stats['cpu_time']
        print('Total time:     {} microseconds'.format(total_time))
        print('Total CPU time: {} microseconds\n'.format(total_time_cpu))

def get_command_line_arguments(argv):
    parameters = []
    arg_name = ''
    arg_value = ''
    for arg in argv[1:]:
        if '=' in arg:
            arg_name, arg_value = arg.split('=')
            parameters.append((arg_name, arg_value))
            arg_name = ''
            arg_value = ''
        else:
          if arg[0] == '-':
              if arg_name is not '':
                parameters.append((arg_name, arg_value))
                arg_value = ''
              arg_name = arg
          else:
              arg_value = arg
    if arg_name is not '':
        parameters.append((arg_name, arg_value))
    return parameters

def update_shapes(shapes, shapes_string: str, inputs_info):
    updated = False
    matches = re.findall(r'(.*?)\[(.*?)\],?', shapes_string)
    if matches:
        for match in matches:
            input_name = match[0]
            parsed_shape = [int(dim) for dim in match[1].split(',')]
            if input_name != '':
                shapes[input_name] = parsed_shape
                updated = True
            else:
                shapes.update({ k:parsed_shape for k in shapes.keys() })
                updated = True
                break
    else:
        raise Exception("Can't parse `shape` parameter: {}".format(shapes_string))
    return updated

def adjust_shapes_batch(shapes, batch_size: int, inputs_info):
    updated = False
    for name, data in inputs_info.items():
        layout = data.input_data.layout
        batch_index = layout.index('N') if 'N' in layout else -1
        if batch_index != -1 and shapes[name][batch_index] != batch_size:
            shapes[name][batch_index] = batch_size
            updated = True
    return updated

def show_available_devices():
    ie = IECore()
    print("\nAvailable target devices:  ", ("  ".join(ie.available_devices)))

def dump_config(filename, config):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename, config):
    with open(filename) as f:
        config.update(json.load(f))
