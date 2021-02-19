import yaml
import argparse
import os
import sys
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("-c","-config", help="path to yml file", type=str)
parser.add_argument("-o","--output", default=os.getcwd(), help="optional: path to where files will be saved default: current working directory", type=str)
args = parser.parse_args()



if args.c:
    config_file = args.c
else: 
    print("config file not passed to entry of script!!  \n ex. -c <path to config>")
    sys.exit(0)
print(args.output)
output_dir = args.output+"/model/FP32/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




print("######PARSING CONFIG FILE {} #######".format(config_file))

with open(config_file, 'r') as file:
    full_config=yaml.full_load(file)
    for files in full_config['files']:
                        
        source_url = files['source']
        name_FP32 = files['name']

        file_name = name_FP32.replace('FP32/','')


        output_file_path = output_dir+file_name

        if not os.path.isfile(output_file_path): 
            new_file = open(output_file_path, "w")
            print("###Downloading File: ", file_name)
            urllib.request.urlretrieve(source_url, output_file_path)
            print("###Download Complete")       
        else:
            print("{} already exist: skipping".format(file_name))


