import torch
import os
import glob
import json
import pickle
from random import random
from tvm import auto_scheduler
from common import (load_and_register_tasks,
                    get_measure_record_filename, get_to_measure_filename)
import threading
import multiprocessing
from tvm.tir.expr import FloatImm
import numpy as np
import random
import argparse
import json

# handle same workload with diff candidates
def handle_file(file_idx, file):
    print(file_idx)  # 打印当前处理的文件索引

    with open(file, 'r') as f:  # 以只读方式打开文件
        lines = f.read().strip().split('\n')  # 读取文件内容，并移除前后的空格和换行符，然后按行分割成列表
    # 使用TVM auto_scheduler的RecordReader读取文件中的记录
    inputs, outputs = auto_scheduler.RecordReader(file).read_lines()
    task = auto_scheduler.measure.recover_measure_input(
        inputs[0]).task  # 使用第一行输入数据恢复度量输入，获取其任务信息

    workloadkey_idx = workloadkey_to_index[task.workload_key[len('["'): len(
        '["6b7583cf23c7c37d3212cad9d06e58c1')]]  # 通过对workload_key进行切片操作，得到workloadkey索引
    workload_args = [int(i) for i in task.workload_key[len(
        '["6b7583cf23c7c37d3212cad9d06e58c1", '): -1].split(', ')]  # 对workload_key进行切片并分割，得到workload的参数，转化为整型

    line_vecs = []  # 初始化line_vecs列表，用于存储每行转化后的向量信息

    min_cost = 1000000  # 初始化最小代价值
    for line_idx, line in enumerate(lines):  # 遍历每一行

        inp = json.loads(line)  # 将每一行的字符串内容转换为json格式
        steps = inp['i'][1][1]  # 获取steps

        step_vecs = []  # 初始化step_vecs列表，用于存储每个步骤的向量信息
        for st in steps:  # 遍历每个步骤
            vec = []  # 初始化vec列表，用于存储当前步骤的向量信息
            # 对当前步骤进行独热编码，并将编码结果追加到vec中
            vec.extend(stepname_to_idx_one_hot[st[0]])

            for i, it in enumerate(st):  # 遍历步骤中的每个元素
                if i == 0:  # 如果是第一个元素，跳过
                    continue
                if isinstance(it, int):  # 如果元素是整型，直接追加到vec中
                    vec.append(it)
                elif isinstance(it, list):  # 如果元素是列表，迭代追加到vec中
                    for ii in it:
                        assert isinstance(ii, int)  # 确认列表中的每个元素都是整型
                        vec.append(ii)
                elif isinstance(it, str):  # 如果元素是字符串，根据不同的条件，将相应的结果追加到vec中
                    if st[0] == 'PR' and 'auto_unroll_max_step' in it:
                        vec.append(auto_unroll_max_step_to_idx[it])
                    elif st[0] == 'CHW':
                        vec.append(chw_dict[it])
                    elif st[0] == 'CHR' and it == 'shared':
                        vec.append(1)
                    else:  # 如果元素不满足任何条件，断言错误
                        assert False
                else:  # 如果元素既不是整型，也不是列表，也不是字符串，断言错误
                    assert False

            assert len(vec) <= max_emb_size  # 断言vec的长度小于等于最大嵌入大小
            # 如果vec的长度小于最大嵌入大小，将0追加到vec中，直到长度等于最大嵌入大小
            for i in range(len(vec), max_emb_size, 1):
                vec.append(0)

            vec = vec[:args.crop_emb_size]  # 对vec进行切片，取出前crop_emb_size个元素
            step_vecs.append(vec)  # 将vec追加到step_vecs中

        assert len(step_vecs) <= max_seq_len  # 断言step_vecs的长度小于等于最大序列长度
        vec = [0] * args.crop_emb_size  # 初始化vec，长度为crop_emb_size，每个元素都是0
        # 如果step_vecs的长度小于最大序列长度，将vec复制并追加到step_vecs中，直到长度等于最大序列长度
        for i in range(len(step_vecs), max_seq_len, 1):
            step_vecs.append(vec.copy())
        # 对step_vecs进行切片，取出前crop_seq_len个元素
        step_vecs = step_vecs[:args.crop_seq_len]

        costs = [x.value for x in outputs[line_idx].costs if isinstance(
            x, FloatImm)]  # 从outputs中提取costs的值，只提取类型为FloatImm的元素
        cost = np.mean(costs)  # 计算costs的平均值
        line_vecs.append((step_vecs, cost))  # 将step_vecs和cost作为元组追加到line_vecs中
        min_cost = min(min_cost, cost)  # 更新最小代价值
        
    line_vecs_new = []  # 初始化line_vecs_new列表，用于存储新的line_vecs
    for line_vec in line_vecs:  # 遍历line_vecs中的每个元素
        step_vecs, cost = line_vec  # 解包每个元素，得到step_vecs和cost
        score = min_cost / cost  # 计算得分，最小代价值除以当前的cost
        # 将step_vecs，得分和最小代价值作为元组追加到line_vecs_new中
        line_vecs_new.append((step_vecs, score, min_cost))
    line_vecs = line_vecs_new  # 更新line_vecs为line_vecs_new

    # 返回结果，包括文件名、文件索引、workloadkey索引、workload_key、workload参数、计算DAG的浮点操作计数和line_vecs
    return (file, file_idx, workloadkey_idx, task.workload_key, workload_args, task.compute_dag.flop_ct, line_vecs)


def make_all_dataset(json_files_path):
    # tasks = load_and_register_tasks()
    json_files = sorted(glob.glob(json_files_path + '/' + '*.json'))
    json_files = random.sample(json_files, args.files_cnt)

    multiprocessing_pool = multiprocessing.Pool()
    que_res_list = []
    # json_files = json_files[1471:]
    for file_idx, file in enumerate(json_files):
        que_res_list.append(multiprocessing_pool.apply_async(
            handle_file, args=(file_idx, file)))
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
    parser.add_argument("--json_files_path", type=str,
                        default='dataset/measure_records/t4')
    parser.add_argument("--files_cnt", type=int, default=2308)
    parser.add_argument("--save_name", type=str, default='')
    parser.add_argument("--platform", type=str, default='cuda')  # or cuda
    parser.add_argument("--crop_seq_len", type=int, default=-1)
    parser.add_argument("--crop_emb_size", type=int, default=-1)
    args = parser.parse_args()

    if args.save_name == '':
        args.save_name = 'tlp_dataset_' + \
            os.path.basename(os.path.normpath(
                args.json_files_path)).replace('-', '_')

    hold_out_tasks = []
    files = [
        'dataset_cpu/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset_cpu/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl',
        'dataset_cpu/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset_cpu/network_info/((bert_base,[(1,128)]),%s).task.pkl',
        'dataset_cpu/network_info/((bert_tiny,[(1,128)]),%s).task.pkl'
    ]
    # for file in files:
    #     tasks_part, task_weights = pickle.load(open(file % args.platform, "rb"))
    #     hold_out_tasks.extend(tasks_part)

    # hold_out_tasks_set = set([task.workload_key for task in hold_out_tasks])

    with open('tlp_make_dataset_str_to_idx_%s.pkl' % args.platform, 'rb') as f:
        workloadkey_to_index, stepname_to_idx, auto_unroll_max_step_to_idx = pickle.load(
            f)

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
