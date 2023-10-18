import torch
from tqdm import tqdm
import wandb

# We use a GFlowNet with the Trajectory Balance (TB) loss
from src.gfn.gflownet import TBGFlowNet
# We use the hyper grid environment
from src.gfn.gym import HyperGrid, MetaScheduleEnv
from src.gfn.modules import DiscretePolicyEstimator
from src.gfn.samplers import Sampler
# NeuralNet is a simple multi-layer perceptron (MLP)
from src.gfn.utils import NeuralNet
from src.gfn.mlc_dataset import *


import IPython
import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.script import tir as T
import pickle

# NOTE: Must import TransformerModule -- pickle cann't find class
from tvm.meta_schedule.cost_model.tlp_cost_model_train import *


if __name__ == "__main__":

    import os
    import sys
    mount_path = "/root"
    root = os.path.join(mount_path, "share/dataset/gflownet_dataset0")

    # ValueError: Object arrays cannot be loaded when allow_pickle=False
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    database_path = "/root/share/dataset/tlp_dataset0"

    info_path = "/root/share/dataset/decode_info"
    gfn_path = "/root/kongdehao/model/gfn"
    # bs = 512
    bs = 16
    dataloader, data_num = gflownet_data_load(
        root, database_path=database_path, num_workers=4, batch_size=bs)

    dataloader0, data_num0 = gflownet_data_load(
        info_path, database_path=info_path, num_workers=4, batch_size=bs)
    # record_iter = iter(record_dataloader)
    train_iter = iter(dataloader)

    train_iter0 = iter(dataloader0)
    x0, workload_paths0, candidate_paths0, decode0, order0, cond0, ptr0, score0 = next(
        train_iter0)
    # epoch = 5000
    epoch = 500
    target = "cuda"

    pbar = tqdm(range(0, epoch))
    pre_f = None
    pre_b = None

    for ep in (pbar):
        cond = None
        ptr = None
        # record, decode, order, last_embedding, run_secs, last_condition, last_ptr_list
        # for step, (decode, order, x, score, cond, ptr) in enumerate(train_iter):
        if True:
            step = ep
            try:
                x, workload_paths, candidate_paths, decode, order, cond, ptr, score = next(
                    train_iter)
            except:
                train_iter = iter(dataloader)
                x, workload_paths, candidate_paths, decode, order, cond, ptr, score = next(
                    train_iter)

            # data_num: 3191
            begin = (step*bs) % data_num
            end = (step*bs+bs) % data_num

            x = x.cuda(non_blocking=True).long()
            # Convert into [batch, -1]
            x = x.view(x.shape[0], -1)

            num = len(workload_paths)
            databases_path = [(workload_paths[i], candidate_paths[i])
                              for i in range(num)]

            num0 = len(workload_paths)
            databases_path0 = [(workload_paths0[i], candidate_paths0[i])
                               for i in range(num0)]
            f_info = (databases_path0, decode0, order0, cond0, ptr0, target)
            b_info = (databases_path, decode, order, cond, ptr, target)

            # NOTE: for test restore_embedding
            info00 = (x0, databases_path0, decode0,
                      order0, cond0, ptr0, target)
            features = restore_embedding(info00)
            print(f"Successful from {step} to {step+bs}!")

    # restore np.load for future normal usage
    np.load = np_load_old
