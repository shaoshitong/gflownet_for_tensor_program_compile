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
    record_path = "/root/share/dataset/tlp_dataset0"
    databases = record_data_load0(record_path)
    print(f"Successful load all record database size = {len(databases)}")
    # for database in databases:
    #     records = database.get_all_tuning_records()
    #     print(len(records))
