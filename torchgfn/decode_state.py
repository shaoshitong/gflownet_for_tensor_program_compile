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


# NOTE: score is runtime, val is variance, tau is temperature factor
def normalize_score(score,
                    _mean=0.003680646535107316,
                    _val=0.0012118761480652196,
                    _min=2.8831801089918256e-06,
                    _max=4.567233072666666,
                    _tau=1):
    score = (score - _mean) / (_val ** (1/2))
    return torch.log(torch.sigmoid(score / _tau))


if __name__ == "__main__":

    from src.gfn.utils.edm_model import mlp_ebm
    # define states len & action len
    state_len = 15 * 1 + 15 * 96
    action_len = 15 * 10 + 15 * 96 * 2 + 1  # add the terminal state
    # 1 - We define the environment and Energy Function
    tlp_path = "/root/kongdehao/model/median_tlp/save_model_v1/tlp_model_14.pkl"
    # tlp_model on device "cuda:7"
    device = "cuda:7"
    with open(tlp_path, 'rb') as f:
        cost_model = pickle.load(f)
    cost_model.to(device)

    # cost_model = torch.load(tlp_path, map_location=device)

    # # NOTE: fake cost model for test
    # Cost Model as discriminator
    # edm_model = mlp_ebm(state_len, 256, 1).cuda()
    # cost_model = edm_model
    # TODO: input TLP in energy, but we need align format
    # Decode result x into tvm, input it into TLP -- torchgfn/mlc_dataset/dataset_embedding/gflownet_embedding.py
    # alpha \in [1, 500]
    env = MetaScheduleEnv(energy=cost_model, alpha=500, device_str=device)

    # 2 - We define the needed modules (neural networks)
    # The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        # We share all the parameters of P_F and P_B, except for the last layer
        torso=module_PF.torso
    )

    # 3 - We define the estimators

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet
    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator,
                     pb=pb_estimator)  # We initialize logZ to 0

    # 5 - We define the sampler and the optimizer

    # We use an on-policy sampler, based on the forward policy
    f_sampler = Sampler(estimator=pf_estimator)

    # We also need a backward policy sampler.
    b_sampler = Sampler(estimator=pb_estimator)

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(
        gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-1})
    # optimizer.add_param_group({"params": edm_model.parameters(), "lr": 1e-3})

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration

    # 7 - Add the Meta-Schedule dataset
    import os
    import sys
    mount_path = "/root"
    root = os.path.join(mount_path, "share/dataset/gflownet_dataset0")

    # ValueError: Object arrays cannot be loaded when allow_pickle=False
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    record_path = "/root/share/dataset/tlp_dataset0"
    # databases = record_data_load(record_path)
    # print(f"Successful load all record database size = {len(databases)}")
    # bs = 512
    bs = 16
    dataloader, data_num = gflownet_data_load(
        root, without_condition=False, num_workers=4, batch_size=bs)
    # record_iter = iter(record_dataloader)
    train_iter = iter(dataloader)
    info_path = "/root/share/dataset/decode_info"
    gfn_path = "/root/kongdehao/model/gfn"
    # epoch = 5000
    epoch = 20
    target = "cuda"

    pbar = tqdm(range(0, epoch))

    for ep in (pbar):
        cond = None
        ptr = None
        # record, decode, order, last_embedding, run_secs, last_condition, last_ptr_list
        # for step, (decode, order, x, score, cond, ptr) in enumerate(train_iter):
        if True:
            step = ep
            try:
                decode, order, x, score, cond, ptr = next(train_iter)
            except:
                train_iter = iter(dataloader)
                decode, order, x, score, cond, ptr = next(train_iter)

            # data_num: 3191
            begin = (step*bs) % data_num
            end = (step*bs+bs) % data_num
            # np.savez(os.path.join(info_path, f'info{step}.npz'), x=x[0], database=databases[begin+0],
            #          decode=decode[0], order=order[0],  cond=cond[0], ptr=ptr[0], target=target)
            # np.savez(os.path.join(info_path, f'info{step}.npz'), decode=decode,
            #          order=order, last_embedding=x, last_condition=cond,
            #          last_ptr_list=ptr, run_secs=score)

            x = x.cuda(non_blocking=True).long()
            # Convert into [batch, -1]
            x = x.view(x.shape[0], -1)
            score = normalize_score(score.cuda(non_blocking=True))
            # create env state
            states = env.States(x)

            workload_paths = [os.path.join(
                record_path, f"workloads_{i}.json") for i in range(begin, end)]
            candidate_paths = [workload_path.replace(
                "workloads", "candidates") for workload_path in workload_paths]
            num = len(candidate_paths)
            # NOTE: not use sorted()
            databases_path = [(workload_paths[i], candidate_paths[i])
                              for i in range(num)]
            # databases = [ms.database.JSONDatabase(
            #     path_workload=workload_paths[i], path_tuning_record=candidate_paths[i]) for i in range(num)]

            npz_file0 = sorted(glob.glob(os.path.join(
                info_path, f"info*.npz"), recursive=True))

            workload_paths0 = [os.path.join(
                info_path, f"workloads_{i}.json") for i in range(0, bs)]

            candidate_paths0 = [workload_path0.replace(
                "workloads", "candidates") for workload_path0 in workload_paths0]
            num0 = len(workload_paths0)
            databases_path0 = [(workload_paths0[i], candidate_paths0[i])
                               for i in range(num0)]
            # databases0 = [ms.database.JSONDatabase(
            #     path_workload=workload_path0[i], path_tuning_record=candidate_path0[i]) for i in range(num0)]

            file0 = npz_file0[0]
            file0 = np.load(file0)

            info0 = formatter(file0)
            decode0, order0, x0, run_secs0, cond0, ptr0 = info0
            f_info = (databases_path0, decode0, order0, cond0, ptr0, target)
            b_info = (databases_path, decode, order, cond, ptr, target)

            # f_infos = (True, f_info)
            # b_infos = (False, b_info)
            # features = restore_embedding(info)
            # print(f"Successful from {step} to {step+bs}!")

            # NOTE: step 1 -- sample trajectory
            f_trajectories = f_sampler.sample_trajectories(
                env=env, n_trajectories=16, info=f_info)
            b_trajectories = b_sampler.sample_trajectories(
                env=env, n_trajectories=16, states=states, info=b_info)
            # TODO: real_y that for training fake cost model -- discriminator
            # real_y = edm_model(x.float())

            # fake_x = f_trajectories.states.tensor[-2,...].clone().detach().float()
            # fake_y = edm_model(fake_x)
            # edm_loss = ((real_y - score) ** 2).mean()
            optimizer.zero_grad()
            # cost model real work place
            # NOTE: step 2 -- compute loss
            f_loss = gfn.loss(env, f_trajectories)
            b_loss = gfn.loss(env, b_trajectories)
            # TODO: remove edm_loss
            loss = b_loss + f_loss  # + edm_loss
            loss.backward()
            optimizer.step()

            if ep % 1 == 0:
                reward = torch.exp(f_trajectories.log_rewards).mean().item()
                pbar.set_postfix(
                    {"b_loss": b_loss.item(), "f_loss": f_loss.item(), "reward": reward})
                # log metrics to wandb
                wandb.log({"b_loss": b_loss.item(),
                          "f_loss": f_loss.item(), "reward": reward})
            if ep & 5 == 0:
                # checkpoint = {"gfn": gfn.state_dict()}
                dir = os.path.join(gfn_path, f"gflownet_{ep}.pth")
                last_dir = os.path.join(gfn_path, f"gflownet_{ep-25}.pth")
                torch.save(gfn, dir)
                os.system(f"rm -rf {last_dir}")

        # save_model_path = "%s/tlp_model_%d.pkl" % (args.save_model_path, epoch)
        # with open(save_model_path, 'wb') as f:
        #     pickle.dump(net.cpu(), f)
        # net.to(device)

    # restore np.load for future normal usage
    np.load = np_load_old
