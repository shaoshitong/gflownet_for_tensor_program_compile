import json,os,sys,random,math
import numpy as np
import glob

if __name__ == "__main__":
    folder_name = "/home/tvm/scripts_tlp/dataset_cpu/measure_records/e5-2673"
    output_folder_name = "/home/tvm/scripts_tlp/dataset_cpu/measure_records/e5-2673-v2"
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    json_files = sorted(glob.glob(folder_name + '/' + '*.json'))
    for json_file in json_files:
        new_lines = []
        with open(json_file, 'r') as f: # 以只读方式打开文件
                lines = f.read().strip().split('\n') # 读取文件内容，并移除前后的空格和换行符，然后按行分割成列表
                for line in lines:
                    new_line = line.replace(" -link-params=0","")
                    new_lines.append(new_line)
        new_lines = '\n'.join(new_lines)
        output_json_file = os.path.join(output_folder_name,json_file.split("/")[-1])
        with open(output_json_file, 'w') as f: # 以只读方式打开文件
             f.write(new_lines)

                