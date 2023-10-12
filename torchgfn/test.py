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
    bs = 4
    mount_path = "/root"
    root = os.path.join(mount_path, "share/dataset/gflownet_dataset0")
    dataloader, data_num = gflownet_data_load(
        root, without_condition=False, num_workers=4, batch_size=bs)

    train_iter = iter(dataloader)

    epoch = 100
    pre = None
    for i in range(epoch):
        try:
            decode, order, x, score, cond, ptr = next(train_iter)
        except:
            train_iter = iter(dataloader)
            decode, order, x, score, cond, ptr = next(train_iter)
        x = x.cuda(non_blocking=True).long()
        # Convert into [batch, -1]
        x = x.view(x.shape[0], -1)
        if pre != None:
            res = torch.nonzero(pre != x)
            print(f"in {i} step, res = {res}")

        pre = x
        # Print the nonzero positions
        # print(nonzero_positions)

        # print(f"x = {x}")
        # print(f"x[0] = {x[0]}")

    # tlp_path = "/root/kongdehao/model/median_tlp/save_model_v1/tlp_model_14.pkl"
    # save_path = "/root/kongdehao/model/median_tlp/save_model_v1/tlp_model_cuda0.pkl"
    # # tlp_model on device "cuda:7"
    # device = "cuda:0"
    # with open(tlp_path, 'rb') as f:
    #     cost_model = pickle.load(f)

    # import torch  # type: ignore
    # from torch import nn
    # # NOTE: specify device id order
    # cost_model = torch.nn.DataParallel(
    #     cost_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    # cost_model.to(device)
    # # with open(save_path, 'wb') as f:
    # #     pickle.dump(best_net.cpu(), f)
    # torch.save(cost_model, save_path)

    # record_path = "/root/share/dataset/tlp_dataset0"
    # databases = record_data_load0(record_path)
    # print(f"Successful load all record database size = {len(databases)}")

    # for database in databases:
    #     records = database.get_all_tuning_records()
    #     print(len(records))
