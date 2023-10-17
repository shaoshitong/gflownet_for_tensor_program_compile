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
    tlp_path = "/root/kongdehao/model/min_tlp/tlp_model_73.pth"

    device = "cuda"
    # with open(tlp_path, 'rb') as f:
    #     cost_model = pickle.load(f)
    # cost_model.to(device)

    cost_model = torch.load(tlp_path, map_location=device)
    cost_model.to(device)

    # # NOTE: fake cost model for test
    # Cost Model as discriminator
    # edm_model = mlp_ebm(state_len, 256, 1).cuda()
    # cost_model = edm_model
    # TODO: input TLP in energy, but we need align format
    # Decode result x into tvm, input it into TLP -- torchgfn/mlc_dataset/dataset_embedding/gflownet_embedding.py
    # alpha \in [1, 500]
    env = MetaScheduleEnv(energy=cost_model, alpha=120, device_str=device)

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
    # gfn.sample_trajectories
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

    database_path = "/root/share/dataset/tlp_dataset0"

    info_path = "/root/share/dataset/0sample_info"
    gfn_path = "/root/kongdehao/model/gfn/forward_gflownet_1480.pth"
    # bs = 512
    bs = 16
    # dataloader, data_num = gflownet_data_load(
    #     root, database_path=database_path, num_workers=4, batch_size=bs)

    dataloader0, data_num0 = gflownet_data_load(
        info_path, database_path=info_path, num_workers=4, batch_size=bs)
    # record_iter = iter(record_dataloader)
    # train_iter = iter(dataloader)

    train_iter0 = iter(dataloader0)
    x0, workload_paths0, candidate_paths0, decode0, order0, cond0, ptr0, score0 = next(
        train_iter0)

    target = "cuda"

    num0 = len(workload_paths0)
    databases_path0 = [(workload_paths0[i], candidate_paths0[i])
                        for i in range(num0)]
    f_info = (databases_path0, decode0, order0, cond0, ptr0, target)


    gfn = torch.load(gfn_path, map_location=device)
    traj =  gfn.sample_trajectories(env=env, n_samples=16, info=f_info)
    xs = traj.states[-2]
    xs = xs.tensor

    import torch.nn.functional as Fun
    MAX_NUMBER = 15
    # (bs, ...)
    databases_path, decodes, orders, conds, ptrs, target = f_info
    bs = xs.shape[0]
    contexts, candidates = [], []
    # print("len of xs", len(xs))
    # TODO:.. print(len)
    old_decision = []
    new_decision = []
    old_ans = 0
    new_ans = 0

    for i in range(bs):
        x, database_path, decode, order, cond, ptr = \
            xs[i], databases_path[i], decodes[i], orders[i], conds[i], ptrs[i]

        cond_x0, cond_y0, cond_x1, cond_y1, max_len, emb0_x, emb1_x = decode
        ex_emb0, ex_emb1 = torch.split(x, [MAX_NUMBER, MAX_NUMBER*96], 0)
        ex_cond0, ex_cond1 = torch.split(
            cond, split_size_or_sections=MAX_NUMBER, dim=0)

        ex_emb1 = ex_emb1.view(MAX_NUMBER, -1)
        # convert into one-hot format
        ex_emb0 = torch.eye(10)[ex_emb0.to(torch.int64)]

        emb0_x = emb0_x.item()
        emb1_x = emb1_x.item()

        emb0, _ = torch.split(ex_emb0, [emb0_x, MAX_NUMBER-emb0_x], 0)
        emb1, _ = torch.split(ex_emb1, [emb1_x, MAX_NUMBER-emb1_x], 0)

        cond0, _ = torch.split(ex_cond0, [cond_x0, MAX_NUMBER-cond_x0], 0)
        cond1, _ = torch.split(ex_cond1, [cond_x1, MAX_NUMBER-cond_x1], 0)

        cond0, _ = torch.split(cond0, [cond_y0, 34-cond_y0], 1)
        cond1, _ = torch.split(cond1, [cond_y1, 34-cond_y1], 1)
        # convert cuda:0 device into cpu
        emb0 = emb0.cpu()
        emb1 = emb1.cpu()
        cond0 = cond0.cpu()
        cond1 = cond1.cpu()

        res = []
        emb_conds = []
        p0 = 0  # emb0 position
        p1 = 0  # emb1
        for i in range(max_len):
            if order[i] == 0:
                res.append(emb0[p0].numpy())
                emb_conds.append(cond0[p0].numpy())
                p0 += 1
            else:
                res.append(emb1[p1].numpy())
                emb_conds.append(cond1[p1].numpy())
                p1 += 1

        if isinstance(ptr, np.ndarray):
            ptr = ptr.astype(int)
            ptr = ptr.tolist()
        else:
            ptr = ptr.int()
            ptr = ptr.tolist()

        workload_path, candidate_path = database_path
        # NOTE: cost 1ms -- not return database, otherwise records is empty list
        database = ms.database.JSONDatabase(path_workload=workload_path,
                                            path_tuning_record=candidate_path)

        records = database.get_all_tuning_records()
        record = records[0]
        candidate = record.as_measure_candidate()
        # results = RunnerResult(run_secs=record.run_secs, error_msg=None)
        # NOTE: cost 10ms
        context = TuneContext(mod=record.workload.mod, target=Target(target))

        sub_sch = candidate.sch
        # trace include instructions & decisions
        sub_trace = sub_sch.trace
        # instructions include deterministic and stochastic
        sub_insts = sub_trace.insts
        # decision only include stochastic instructions
        sub_decisions = sub_trace.decisions
        old_sch = candidate.sch
        # NOTE: Double Bug, must add target = "cuda", target = target. NOT (sch.mod, target)
        lib = tvm.build(old_sch.mod, target="cuda")
        # lib = tvm.build(sch.mod, target=target)
        a_nd = tvm.nd.array(np.random.uniform(size=(1, 64, 768)).astype("float32"), device=tvm.cuda())
        b_nd = tvm.nd.array(np.random.uniform(size=(1, 64, 1)).astype("float32"), device=tvm.cuda())

        f_timer_after = lib.time_evaluator("main", tvm.cuda())
        # print(f"{record.workload.mod}")
        tmp = f_timer_after(a_nd, b_nd).mean * 1000
        old_ans += tmp
        print("Time cost of MyModule before tuning: %.3f ms" % (tmp))
        
        # print(f"after record = {record}, candidate = {candidate}, decision = ")
        gm = GflowNetEmbedding()
        new_sub_insts, new_sub_decisions = gm(sub_insts, sub_decisions, False, embedding_results=res,
                                              embedding_conditions=emb_conds, count_Ptr_results=ptr)

        # NOTE: new decision is null list -- gm must pass valid insts & decisions
        # print(f"new decision = {new_sub_decisions}")

        # Must use with_decision() to set sub_trace
        for new_sub_inst, new_sub_decision in zip(new_sub_insts, new_sub_decisions):
            # new_sub_decision = tvm.tir.const(1, dtype='int32')
            # NOTE: bug1 must assign to sub_trace
            sub_trace = sub_trace.with_decision(
                new_sub_inst, new_sub_decision, True)
        nn = len(list(sub_trace.decisions.values()))
        # if nn > 2:
        #     print(f"Encounter new condition in {candidate_path}!")
        #     print(f"old decision = {list(sub_decisions.values())}")
        #     print(f"new decisions = {new_sub_decisions}")
        #     # print(f"res = {res}")
        old_decision.append(list(sub_decisions.values()))
        new_decision.append(new_sub_decisions)

        from tvm.meta_schedule.database.database import TuningRecord

        new_database = database
        new_database.commit_workload(record.workload.mod)
        new_database.commit_tuning_record(TuningRecord(
            sub_trace,
            record.workload,
            record.run_secs,
            Target(target),
            candidate.args_info))

        records = new_database.get_all_tuning_records()
        # print("records shape = ", len(records))
        # NOTE: commit add new candidates in json
        record = records[-1]
        # NOTE: n=3, but decision=4
        candidate = record.as_measure_candidate()
        sch = candidate.sch
        context = TuneContext(mod=record.workload.mod, target=Target(target))
        # TODO: check same context
        # if len(contexts) > 0:
        #     if context != contexts[-1]:
        #         print("diff context")

        # check correctness for
        # print(f"final candidate decision = {list(sub_decisions.values())}")
        contexts.append(context)
        candidates.append(candidate)
        
        # NOTE: Double Bug, must add target = "cuda", target = target. NOT (sch.mod, target)
        lib = tvm.build(sch.mod, target="cuda")
        # lib = tvm.build(sch.mod, target=target)
        a_nd = tvm.nd.array(np.random.uniform(size=(1, 64, 768)).astype("float32"), device=tvm.cuda())
        b_nd = tvm.nd.array(np.random.uniform(size=(1, 64, 1)).astype("float32"), device=tvm.cuda())
        c_nd = tvm.nd.empty((128, 128), "float32", device = tvm.cuda())
        f_timer_after = lib.time_evaluator("main", tvm.cuda())
        # print(f"{record.workload.mod}")
        tmp = f_timer_after(a_nd, b_nd).mean * 1000
        new_ans += tmp
        print("Time cost of MyModule after tuning: %.3f ms" % (tmp))


    print(f"old mean time = {old_ans*1.0/bs} ms")
    print(f"new mean time = {new_ans*1.0/bs} ms")
    print(f"Finish")
    # for ep in (pbar):
    #     cond = None
    #     ptr = None
    #     # record, decode, order, last_embedding, run_secs, last_condition, last_ptr_list
    #     # for step, (decode, order, x, score, cond, ptr) in enumerate(train_iter):
    #     if True:
    #         step = ep
    #         try:
    #             x, workload_paths, candidate_paths, decode, order, cond, ptr, score = next(
    #                 train_iter)
    #         except:
    #             train_iter = iter(dataloader)
    #             x, workload_paths, candidate_paths, decode, order, cond, ptr, score = next(
    #                 train_iter)

    #         # data_num: 3191
    #         begin = (step*bs) % data_num
    #         end = (step*bs+bs) % data_num

    #         x = x.cuda(non_blocking=True).long()
    #         # Convert into [batch, -1]
    #         x = x.view(x.shape[0], -1)

    #         score = normalize_score(score.cuda(non_blocking=True))
    #         if torch.all(x == 0):
    #             print("x is all zeros")
    #         # create env state
    #         states = env.States(x)

    #         # for i in range(decode0.shape[0]):

    #         #     workload_path, candidate_path = workload_paths0[i], candidate_paths0[i]
    #         #     # NOTE: cost 1ms -- not return database, otherwise records is empty list
    #         #     database = ms.database.JSONDatabase(path_workload=workload_path,
    #         #                                 path_tuning_record=candidate_path)
    #         #     records = database.get_all_tuning_records()
    #         #     record = records[0]
    #         #     candidate = record.as_measure_candidate()
    #         #     # results = RunnerResult(run_secs=record.run_secs, error_msg=None)
    #         #     # NOTE: cost 10ms
    #         #     context = TuneContext(mod=record.workload.mod, target=Target(target))

    #         #     sub_sch = candidate.sch
    #         #     # trace include instructions & decisions
    #         #     sub_trace = sub_sch.trace
    #         #     # instructions include deterministic and stochastic
    #         #     sub_insts = sub_trace.insts
    #         #     # decision only include stochastic instructions
    #         #     sub_decisions = sub_trace.decisions
    #         #     print(f"old decision = {list(sub_decisions.values())}")
    #         #     cond_x0, cond_y0, cond_x1, cond_y1, max_len, emb0_x, emb1_x = decode0[0]
    #         #     print(f"{cond_x0, cond_y0, cond_x1, cond_y1}")

    #         num = len(workload_paths)
    #         databases_path = [(workload_paths[i], candidate_paths[i])
    #                           for i in range(num)]

    #         num0 = len(workload_paths)
    #         databases_path0 = [(workload_paths0[i], candidate_paths0[i])
    #                            for i in range(num0)]
    #         f_info = (databases_path0, decode0, order0, cond0, ptr0, target)
    #         b_info = (databases_path, decode, order, cond, ptr, target)

    #         # # NOTE: for test restore_embedding
    #         # info00 = (x0, databases_path0, decode0,
    #         #           order0, cond0, ptr0, target)
    #         # features = restore_embedding(info00)
    #         # print(f"Successful from {step} to {step+bs}!")

    #         # NOTE: step 1 -- sample trajectory
    #         f_trajectories = f_sampler.sample_trajectories(
    #             env=env, n_trajectories=16, info=f_info)
    #         # print(
    #         #     f"Sample forward trajectory! reward = {torch.exp(f_trajectories.log_rewards).mean().item()}")
    #         # b_trajectories = b_sampler.sample_trajectories(
    #         #     env=env, n_trajectories=16, states=states, info=b_info)
    #         # print(
    #         #     f"Sample backward trajectory! reward = {torch.exp(b_trajectories.log_rewards).mean().item()}")

    #         # pre_f = f_trajectories.features
    #         # pre_b = b_trajectories.features
    #         # TODO: real_y that for training fake cost model -- discriminator
    #         # real_y = edm_model(x.float())

    #         # fake_x = f_trajectories.states.tensor[-2,...].clone().detach().float()
    #         # fake_y = edm_model(fake_x)
    #         # edm_loss = ((real_y - score) ** 2).mean()
    #         optimizer.zero_grad()
    #         # cost model real work place
    #         # NOTE: step 2 -- compute loss
    #         f_loss = gfn.loss(env, f_trajectories)
    #         # b_loss = gfn.loss(env, b_trajectories)
    #         # print(
    #         #     f"After loss, forward reward = {torch.exp(f_trajectories.log_rewards).mean().item()}")
    #         # print(
    #         #     f"After loss, backward reward = {torch.exp(b_trajectories.log_rewards).mean().item()}")
    #         # TODO: remove edm_loss
    #         loss = f_loss  # b_loss + edm_loss
    #         loss.backward()
    #         optimizer.step()

    #         if ep % 1 == 0:
    #             reward = torch.exp(f_trajectories.log_rewards).mean().item()
    #             pbar.set_postfix(
    #                 {"f_loss": f_loss.item(), "reward": reward})
    #             # # log metrics to wandb
    #             # wandb.log({"res[0]": f_trajectories.res[0], "res[1]": f_trajectories.res[1],
    #             #            "res[2]": f_trajectories.res[2], "f_loss": f_loss.item(), "reward": reward})
    #         if ep % 5 == 0:
    #             # checkpoint = {"gfn": gfn.state_dict()}
    #             dir = os.path.join(gfn_path, f"forward_gflownet_{ep}.pth")
    #             last_dir = os.path.join(
    #                 gfn_path, f"forward_gflownet_{ep-20}.pth")
    #             torch.save(gfn, dir)
    #             os.system(f"rm -rf {last_dir}")

    #     # save_model_path = "%s/tlp_model_%d.pkl" % (args.save_model_path, epoch)
    #     # with open(save_model_path, 'wb') as f:
    #     #     pickle.dump(net.cpu(), f)
    #     # net.to(device)

    # restore np.load for future normal usage
    np.load = np_load_old
