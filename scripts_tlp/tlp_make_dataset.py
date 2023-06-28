import torch
import os
import glob
import json
import pickle
from random import random
from tvm import auto_scheduler
from common import (load_and_register_tasks, get_measure_record_filename, get_to_measure_filename)
import threading
import multiprocessing
from tvm.tir.expr import FloatImm
import numpy as np
import random
import argparse

def handle_file(file_idx, file):
    print(file_idx) # 打印当前处理的文件索引

    with open(file, 'r') as f: # 以只读方式打开文件
        lines = f.read().strip().split('\n') # 读取文件内容，并移除前后的空格和换行符，然后按行分割成列表

    inputs, outputs = auto_scheduler.RecordReader(file).read_lines() # 使用TVM auto_scheduler的RecordReader读取文件中的记录
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task # 使用第一行输入数据恢复度量输入，获取其任务信息

    workloadkey_idx = workloadkey_to_index[task.workload_key[len('["'): len('["6b7583cf23c7c37d3212cad9d06e58c1')]] # 通过对workload_key进行切片操作，得到workloadkey索引
    workload_args = [int(i) for i in task.workload_key[len('["6b7583cf23c7c37d3212cad9d06e58c1", '): -1].split(', ')] # 对workload_key进行切片并分割，得到workload的参数，转化为整型

    line_vecs = [] # 初始化line_vecs列表，用于存储每行转化后的向量信息

    min_cost = 1000000 # 初始化最小代价值
    for line_idx, line in enumerate(lines): # 遍历每一行

        inp = json.loads(line) # 将每一行的字符串内容转换为json格式
        steps = inp['i'][1][1] # 获取steps

        step_vecs = [] # 初始化step_vecs列表，用于存储每个步骤的向量信息
        for st in steps: # 遍历每个步骤
            vec = [] # 初始化vec列表，用于存储当前步骤的向量信息
            vec.extend(stepname_to_idx_one_hot[st[0]]) # 对当前步骤进行独热编码，并将编码结果追加到vec中

            for i, it in enumerate(st): # 遍历步骤中的每个元素
                if i == 0: # 如果是第一个元素，跳过
                    continue
                if isinstance(it, int): # 如果元素是整型，直接追加到vec中
                    vec.append(it)
                elif isinstance(it, list): # 如果元素是列表，迭代追加到vec中
                    for ii in it:
                        assert isinstance(ii, int) # 确认列表中的每个元素都是整型
                        vec.append(ii)
                elif isinstance(it, str): # 如果元素是字符串，根据不同的条件，将相应的结果追加到vec中
                    if st[0] == 'PR' and 'auto_unroll_max_step' in it:
                        vec.append(auto_unroll_max_step_to_idx[it])
                    elif st



def make_all_dataset(json_files_path):
    tasks = load_and_register_tasks()
    json_files = sorted(glob.glob(json_files_path + '/' + '*.json'))
    json_files = random.sample(json_files, args.files_cnt)

    multiprocessing_pool = multiprocessing.Pool()
    que_res_list = []
    # json_files = json_files[1471:]
    for file_idx, file in enumerate(json_files):
        que_res_list.append(multiprocessing_pool.apply_async(handle_file, args=(file_idx, file)))
        # handle_file(file_idx, file)

    multiprocessing_pool.close()
    multiprocessing_pool.join()

    file_vecs = []
    for que_res in que_res_list:
        file_vecs.append(que_res.get())

    return file_vecs


def split_dataset(file_vecs):
    train_and_val_dataset = []
    test_data = []

    for file_vec_idx, file_vec in enumerate(file_vecs):

        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = file_vec
        print(file_vec_idx, len(line_vecs))

        if workloadkey in hold_out_tasks_set:
            test_data.append(file_vec)
        else:
            train_and_val_dataset.append(file_vec)

    train_and_val_dataset_new = []
    for data_idx, data in enumerate(train_and_val_dataset):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        train_and_val_dataset_new.extend(line_vecs)
        # for line_index, line in enumerate(line_vecs):
        #     train_and_val_dataset_new.append([file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line, line_index])


    with open(f'{args.save_name}_{args.files_cnt}_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(f'{args.save_name}_{args.files_cnt}_train_and_val.pkl', 'wb') as f:
        pickle.dump(train_and_val_dataset_new, f)



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files_path", type=str, default='dataset/measure_records/platinum-8272')
    parser.add_argument("--files_cnt", type=int, default=2308)
    parser.add_argument("--save_name", type=str, default='')
    parser.add_argument("--platform", type=str, default='llvm')  # or cuda
    parser.add_argument("--crop_seq_len", type=int, default=-1)
    parser.add_argument("--crop_emb_size", type=int, default=-1)
    args = parser.parse_args()

    if args.save_name == '':
        args.save_name = 'tlp_dataset_' + os.path.basename(os.path.normpath(args.json_files_path)).replace('-', '_')

    hold_out_tasks = []
    files = [
        'dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl',
        'dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl'
    ]
    for file in files:
        tasks_part, task_weights = pickle.load(open(file % args.platform, "rb"))
        hold_out_tasks.extend(tasks_part)

    hold_out_tasks_set = set([task.workload_key for task in hold_out_tasks])

    with open('tlp_make_dataset_str_to_idx_%s.pkl' % args.platform, 'rb') as f:
        workloadkey_to_index, stepname_to_idx, auto_unroll_max_step_to_idx = pickle.load(f)


    stepname_to_idx_one_hot = {}
    for key, value in stepname_to_idx.items():
        one_hot = [0] * 11
        one_hot[stepname_to_idx[key]-1] = 1
        stepname_to_idx_one_hot[key] = one_hot

    chw_dict = {
        'local': 1,
        'shared': 2,
        'global': 3,
    }

    if args.platform == 'llvm':
        max_seq_len = 54
        max_emb_size = 40
    else:
        max_seq_len = 69
        max_emb_size = 49

    if args.crop_seq_len == -1 or args.crop_emb_size == -1:
        if args.platform == 'llvm':
            args.crop_seq_len = 25
            args.crop_emb_size = 22
        else:
            args.crop_seq_len = 40
            args.crop_emb_size = 20

    print(args)

    file_vecs = make_all_dataset(args.json_files_path)
    split_dataset(file_vecs)
    print('make dataset tlp done.')
