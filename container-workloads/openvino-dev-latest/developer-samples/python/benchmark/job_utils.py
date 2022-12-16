import json
from os import replace
import os
import time
import pandas as pd

def format_key(device_name):
    key = device_name.replace('GPU','').replace('CPU','').replace('VPU','').replace('Â','').replace('â','').replace('¢','').replace('®','')
    return key.replace('/','').replace('\n','').replace('Intel','').replace('®','').replace('™','').replace('„','')

def create_jobs_from_config(api):
    job_config = []

    with open("job_config.json","r") as jfile:
        job_config = json.load(jfile)

    results_base_path = job_config["job"]["results_path"]    
    cmd = job_config['job']['command']

    device_dict={}
    cmds = []
    keys = []
    def append_qsub_command(node, results_suffix, target, api):
        keys.append(results_suffix)
        results = results_base_path + results_suffix
        cmds.append(cmd.replace('NODE',node).replace('RESULTS', results.replace(' ','_')).replace('TARGET', target).replace('API',api))

    for input in job_config['inputs']:
        if 'Target_node' in input:
            for system in input['Target_node']['options']:
                if system['name'] != 'Select Node':
                    node_id = system['defines']['NODE']
                    node_name = system["name"]
                    node_key = format_key(node_name).strip()
                    
                    for device in system['controls']['Target_arch']:
                        device_key = format_key(device).strip()                
                        if 'GPU' in device:
                            append_qsub_command(node_id, node_key +" " + device_key, 'GPU', api)
                        elif 'VPU' in device:
                            append_qsub_command(node_id, node_key + " " + device_key, 'HDDL', api)                          
                        elif 'CPU' in device:
                            append_qsub_command(node_id, node_key + " " + device_key, 'CPU', api)
    return keys, cmds

def wait_for_job_to_finish(job_id):
    
    print(job_id[0]) 
    if job_id:
        
        print("Job submitted to the queue. Waiting for it to complete .", end="")
        filename = "benchmark_filename_{}.txt".format(job_id[0].split(".")[0])
        
        while not os.path.exists(filename):  # Wait until the file report is created.
            time.sleep(1)
            print(".", end="")
        
        # Print the results
        with open(filename) as f:
            results_dir = f.read().split("\n")[0]
            
        report_filename = os.path.join('./results/', job_id[0].split(".")[0], "benchmark_report.csv") # Wait until the file report is created.
        timeout = 600
        timetaken = 0
        while not os.path.exists(report_filename) and timeout > timetaken:
            time.sleep(1)
            print(".", end="")
            timetaken += 1
        
        if not os.path.exists(report_filename):
            print("Job timed out")
            return None
        
        df = pd.read_csv(report_filename, delimiter=";")
        
        try:
            throughput = float(df.loc["throughput"][0])
            device = df.loc["target device"][0]
            load_time = float(df.loc["load network time (ms)"][0])
            read_time = float(df.loc["read network time (ms)"][0])
        except:
            print("Error in output, please see results.")
            return None     
        
        os.remove(filename) # Cleanup
        
    else:
        print("Error in job submission.")
        
        throughput = None
        device = None
        load_time = None
        read_time = None
        
    return {"Throughput (FPS)": throughput, 
            "Load network time (ms)" : load_time,
            "Read network time (ms)" : read_time}