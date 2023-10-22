import glob
from .dataset_embedding import GflowNetEmbedding, load_all_files, load_workload_and_candidate
import tvm
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tvm.meta_schedule.cost_model.tlp_cost_model_train import from_json, load_data
# from .dataset_embedding import GflowNetEmbedding, load_all_files
from tvm import meta_schedule as ms
from tvm.ir import load_json

import tvm
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule import TuneContext, FeatureExtractor, MeasureCandidate
from tvm.target import Target
from tvm.meta_schedule.feature_extractor import PerStoreFeature
from tvm.runtime import NDArray
from tvm.meta_schedule.utils import shash2hex
import numpy as np
from tvm.runtime import NDArray
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.tune_context import TuneContext
from typing import Dict, List, NamedTuple, Optional, Tuple
from tvm.meta_schedule.cost_model.tlp_cost_model_train import *
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import sys
# Add the parent directory of mypackage to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import subpackage.submodule
# from mypackage.subpackage import submodule

# To make a GFlowNet dataset
# TODO: need for add workload(in context) info as condition


class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                                          None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


def compute_rankloss():

    def read_specific_line(filename, line_number):
        # 打开文件并读取特定行的内容
        with open(filename, 'r') as file:
            lines = file.readlines()
            specific_line = lines[line_number - 1]  # 行号从1开始，所以需要减1
            # 使用split方法将字符串拆分为两部分
        parts = specific_line.split(' ', 1)
        # print(f"parts[1] = {parts[1]}")
        # 提取数字并转换为ndarray类型
        numbers = torch.tensor([float(num)
                                for num in parts[1].split('  ') if num.strip()]).to("cuda")

        return numbers

    # 读取文件中的第3行，并将其中的数字转换为ndarray类型
    filename = '/root/kongdehao/model/0test_tlp/0records_median_run_8.txt'
    line_number = 17
    tlp_old_14 = read_specific_line(filename, line_number)

    line_number = 16
    tlp_median_home0_13 = read_specific_line(filename, line_number)

    line_number = 15
    tlp_median_14 = read_specific_line(filename, line_number)

    line_number = 14
    tlp_median_home_14 = read_specific_line(filename, line_number)

    line_number = 13
    hardware = read_specific_line(filename, line_number)
    device = "cuda"
    loss_func = LambdaRankLoss(device)
    loss_tlp_old_14 = loss_func(tlp_old_14, hardware)

    loss_tlp_median_home0_13 = loss_func(tlp_median_home0_13, hardware)

    loss_tlp_median_14 = loss_func(tlp_median_14, hardware)

    loss_tlp_median_home_14 = loss_func(tlp_median_home_14, hardware)

    save_path = "/root/kongdehao/model/0test_tlp"
    with open(os.path.join(save_path, f'0records_rank_loss_4.txt'), 'w') as file:
        # file.write("diff hardware from diff_tlp_median_57_time mean = " +
        #            str(np.mean(diff_tlp_median_57_time)) + "  var = " + str(np.var(diff_tlp_median_57_time)) + '\n')
        # file.write("diff hardware from diff_tlp_min_77_time mean = " +
        #            str(np.mean(diff_tlp_min_77_time)) + "  var = " + str(np.var(diff_tlp_min_77_time)) + '\n')
        # file.write("diff hardware from diff_tlp_v2_median_69_time mean = " +
        #            str(np.mean(diff_tlp_v2_median_69_time)) + "  var = " + str(np.var(diff_tlp_v2_median_69_time)) + '\n')
        # file.write("diff hardware from diff_tlp_v2_min_44_time mean = " +
        #            str(np.mean(diff_tlp_v2_min_44_time)) + "  var = " + str(np.var(diff_tlp_v2_min_44_time)) + '\n')

        file.write("Rank Loss hardware from tlp_median_home_14_time = " +
                   str(loss_tlp_median_home_14) + '\n')
        file.write("Rank Loss hardware from tlp_median_14_time = " +
                   str(loss_tlp_median_14) + '\n')
        file.write("Rank Loss hardware from tlp_median_home0_13_time = " +
                   str(loss_tlp_median_home0_13) + '\n')
        file.write("Rank Loss hardware from tlp_old_14_time = " +
                   str(loss_tlp_old_14) + '\n')
    print(f"Finish!!!")


def measure_tlp(data_path, save_path):
    assert os.path.exists(data_path), f"{data_path} not exists!"
    # database include candidates(trace, instructions&decision) & workload(subgraph)
    databases = load_all_files(data_path)

    print("Successfully Load Databases!")

    device = "cuda"

    tlp_median_57_path = "/root/kongdehao/model/0test_tlp/tlp_median_57.pth"
    tlp_median_57 = torch.load(tlp_median_57_path, map_location=device)
    tlp_median_57.to(device)

    tlp_min_77_path = "/root/kongdehao/model/0test_tlp/tlp_min_77.pth"
    tlp_min_77 = torch.load(tlp_min_77_path, map_location=device)
    tlp_min_77.to(device)

    tlp_v2_median_69_path = "/root/kongdehao/model/0test_tlp/tlp_v2_median_69.pth"
    tlp_v2_median_69 = torch.load(tlp_v2_median_69_path, map_location=device)
    tlp_v2_median_69.to(device)

    tlp_v2_min_44_path = "/root/kongdehao/model/0test_tlp/tlp_v2_min_44.pth"
    tlp_v2_min_44 = torch.load(tlp_v2_min_44_path, map_location=device)
    tlp_v2_min_44.to(device)

    tlp_median_home_14_path = "/root/kongdehao/model/tlp/median/tlp_median_home_14.pth"

    tlp_median_home_14 = torch.load(
        tlp_median_home_14_path, map_location=device)
    tlp_median_home_14.to(device)

    tlp_median_14_path = "/root/kongdehao/model/tlp/median/tlp_median_14.pth"
    tlp_median_14 = torch.load(tlp_median_14_path, map_location=device)
    tlp_median_14.to(device)

    tlp_median_home0_13_path = "/root/kongdehao/model/tlp/median/tlp_median_home0_13.pth"
    tlp_median_home0_13 = torch.load(
        tlp_median_home0_13_path, map_location=device)
    tlp_median_home0_13.to(device)

    tlp_old_14_path = "/root/kongdehao/model/median_tlp/save_model_v1/tlp_model_14.pkl"
    with open(tlp_old_14_path, 'rb') as f:
        tlp_old_14 = pickle.load(f)
    tlp_old_14.to(device)
    # Modify the device_ids
    tlp_old_14.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # Modify the src_device_obj
    tlp_old_14.src_device_obj = torch.device('cuda:0')
    # Modify the output_device
    tlp_old_14.output_device = torch.device('cuda:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get min cost time from multiple measure runtime
    def _min_cost(res) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.min([float(s) for s in res.run_secs]))

    # get median cost time from multiple measure runtime
    def _median_cost(res) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.median([float(s) for s in res.run_secs]))

    hardware_time = []
    tlp_median_home_14_time = []
    tlp_median_14_time = []
    tlp_median_home0_13_time = []
    tlp_old_14_time = []
    tlp_median_57_time = []
    tlp_min_77_time = []
    tlp_v2_median_69_time = []
    tlp_v2_min_44_time = []
    import wandb
    wandb_project = "Test various TLP performance with Rank Loss"
    use_wandb = len(wandb_project) > 0
    if use_wandb:
        wandb.init(project=wandb_project)
        wandb.config.update({
            "learning_rate": 1e-3,
            "architecture": "EBM GFlowNet",
            "dataset": "GFlowNet Dataset",
        })

    target = "cuda"
    counter = 0
    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        for record in records:
            # convert record into measured candidates
            # measure_candidate for
            sub_sch = record.as_measure_candidate().sch
            # record.workload is workload info
            min_cost = _min_cost(record)
            # min_cost = _median_cost(record)

            # NOTE: pass invalid run_sec
            if min_cost == 1e10:
                print(f"pass invalid run_sec in {counter}")
                continue

            hardware_time.append(min_cost)
            wandb.log({"hardware_time": min_cost})
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions

            # NOTE: n=3, but decision=4
            candidate = record.as_measure_candidate()

            context = TuneContext(mod=record.workload.mod,
                                  target=Target(target), num_threads=112)

            features, _ = extract_features(context, [candidate])
            val_dataloader = SegmentDataloder_new(
                features, shuffle=False, batch_size=1
            )

            for batch_data, _ in val_dataloader:
                batch_data = batch_data.to(device)
                tlp_median_home_14_time.append(
                    tlp_median_home_14(batch_data).item())
                tlp_median_14_time.append(tlp_median_14(batch_data).item())
                tlp_median_home0_13_time.append(
                    tlp_median_home0_13(batch_data).item())
                tlp_old_14_time.append(tlp_old_14(batch_data).item())

                tlp_median_57_time.append(tlp_median_57(batch_data).item())
                tlp_min_77_time.append(tlp_min_77(batch_data).item())
                tlp_v2_median_69_time.append(
                    tlp_v2_median_69(batch_data).item())
                tlp_v2_min_44_time.append(tlp_v2_min_44(batch_data).item())
                # log metrics to wandb
                wandb.log({"tlp_median_home_14_time": tlp_median_home_14_time[-1],
                           "tlp_median_14_time": tlp_median_14_time[-1],
                           "tlp_median_home0_13_time": tlp_median_home0_13_time[-1],
                           "tlp_old_14_time": tlp_old_14_time[-1],
                           "tlp_median_57_time": tlp_median_57_time[-1],
                           "tlp_min_77_time": tlp_min_77_time[-1],
                           "tlp_v2_median_69_time": tlp_v2_median_69_time[-1],
                           "tlp_v2_min_44_time": tlp_v2_min_44_time[-1]})

            print(f"Finish counter = {counter}")
            counter += 1

    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    hardware = torch.tensor(hardware_time).to(device)
    tlp_median_57_time = torch.tensor(tlp_median_57_time).to(device)
    tlp_min_77_time = torch.tensor(tlp_min_77_time).to(device)
    tlp_v2_median_69_time = torch.tensor(tlp_v2_median_69_time).to(device)
    tlp_v2_min_44_time = torch.tensor(tlp_v2_min_44_time).to(device)
    tlp_median_home_14_time = torch.tensor(tlp_median_home_14_time).to(device)
    tlp_median_14_time = torch.tensor(tlp_median_14_time).to(device)
    tlp_median_home0_13_time = torch.tensor(
        tlp_median_home0_13_time).to(device)
    tlp_old_14_time = torch.tensor(tlp_old_14_time).to(device)

    loss_func = LambdaRankLoss(device)
    loss_median_57 = loss_func(tlp_median_57_time, hardware)
    loss_min_77 = loss_func(tlp_min_77_time, hardware)
    loss_v2_median_69 = loss_func(tlp_v2_median_69_time, hardware)
    loss__v2_min_44 = loss_func(tlp_v2_min_44_time, hardware)

    loss_tlp_old_14 = loss_func(tlp_old_14_time, hardware)

    loss_tlp_median_home0_13 = loss_func(tlp_median_home0_13_time, hardware)

    loss_tlp_median_14 = loss_func(tlp_median_14_time, hardware)

    loss_tlp_median_home_14 = loss_func(tlp_median_home_14_time, hardware)

    wandb.log({"loss_median_57": loss_median_57, "loss_min_77": loss_min_77,
               "loss_v2_median_69": loss_v2_median_69, "loss__v2_min_44": loss__v2_min_44,
               "loss_tlp_old_14": loss_tlp_old_14, "loss_tlp_median_home0_13": loss_tlp_median_home0_13,
               "loss_tlp_median_14": loss_tlp_median_14, "loss_tlp_median_home_14": loss_tlp_median_home_14})

    save_path = "/root/kongdehao/model/0test_tlp"
    with open(os.path.join(save_path, f'0records_part_rank_loss_8.txt'), 'w') as file:
        file.write("Rank Loss  hardware from diff_tlp_median_57_time = " +
                   str(loss_median_57) + '\n')
        file.write("Rank Loss hardware from diff_tlp_min_77_time = " +
                   str(loss_min_77) + '\n')
        file.write("Rank Loss hardware from diff_tlp_v2_median_69_time = " +
                   str(loss_v2_median_69) + '\n')
        file.write("Rank Loss hardware from diff_tlp_v2_min_44_time = " +
                   str(loss__v2_min_44) + '\n')

        file.write("Rank Loss hardware from tlp_median_home_14_time = " +
                   str(loss_tlp_median_home_14) + '\n')
        file.write("Rank Loss hardware from tlp_median_14_time = " +
                   str(loss_tlp_median_14) + '\n')
        file.write("Rank Loss hardware from tlp_median_home0_13_time = " +
                   str(loss_tlp_median_home0_13) + '\n')
        file.write("Rank Loss hardware from tlp_old_14_time = " +
                   str(loss_tlp_old_14) + '\n')
    print(f"Finish!!!")
    # hardware_time = normalize(np.array(hardware_time))
    # tlp_median_57_time = normalize(np.array(tlp_median_57_time))
    # diff_tlp_median_57_time = hardware_time - tlp_median_57_time

    # hardware_time = normalize(np.array(hardware_time))
    # tlp_min_77_time = normalize(np.array(tlp_min_77_time))
    # diff_tlp_min_77_time = hardware_time - tlp_min_77_time

    # hardware_time = normalize(np.array(hardware_time))
    # tlp_v2_median_69_time = normalize(np.array(tlp_v2_median_69_time))
    # diff_tlp_v2_median_69_time = hardware_time - tlp_v2_median_69_time

    # hardware_time = normalize(np.array(hardware_time))
    # tlp_v2_min_44_time = normalize(np.array(tlp_v2_min_44_time))
    # diff_tlp_v2_min_44_time = hardware_time - tlp_v2_min_44_time

    # hardware_time = normalize(np.array(hardware_time))
    # tlp_median_home_14_time = normalize(np.array(tlp_median_home_14_time))
    # diff_home_14 = hardware_time - tlp_median_home_14_time

    # tlp_median_14_time = normalize(np.array(tlp_median_14_time))
    # diff_median_14 = hardware_time - tlp_median_14_time
    # tlp_median_home0_13_time = normalize(np.array(tlp_median_home0_13_time))
    # diff_home0_13 = hardware_time - tlp_median_home0_13_time
    # tlp_old_14_time = normalize(np.array(tlp_old_14_time))
    # diff_old_14 = hardware_time - tlp_old_14_time

    # with open(os.path.join(save_path, f'0records_median_run_8.txt'), 'w') as file:
    #     file.write("diff hardware from diff_tlp_median_57_time mean = " +
    #                str(np.mean(diff_tlp_median_57_time)) + "  var = " + str(np.var(diff_tlp_median_57_time)) + '\n')
    #     file.write("diff hardware from diff_tlp_min_77_time mean = " +
    #                str(np.mean(diff_tlp_min_77_time)) + "  var = " + str(np.var(diff_tlp_min_77_time)) + '\n')
    #     file.write("diff hardware from diff_tlp_v2_median_69_time mean = " +
    #                str(np.mean(diff_tlp_v2_median_69_time)) + "  var = " + str(np.var(diff_tlp_v2_median_69_time)) + '\n')
    #     file.write("diff hardware from diff_tlp_v2_min_44_time mean = " +
    #                str(np.mean(diff_tlp_v2_min_44_time)) + "  var = " + str(np.var(diff_tlp_v2_min_44_time)) + '\n')

    #     file.write("diff hardware from tlp_median_home_14_time mean = " +
    #                str(np.mean(diff_home_14)) + "  var = " + str(np.var(diff_home_14)) + '\n')
    #     file.write("diff hardware from tlp_median_14_time mean = " +
    #                str(np.mean(diff_median_14)) + "  var = " + str(np.var(diff_median_14)) + '\n')
    #     file.write("diff hardware from tlp_median_home0_13_time mean = " +
    #                str(np.mean(diff_home0_13)) + "  var = " + str(np.var(diff_home0_13)) + '\n')
    #     file.write("diff hardware from tlp_old_14_time mean = " +
    #                str(np.mean(diff_old_14)) + "  var = " + str(np.var(diff_old_14)) + '\n')
    #     # file.write("diff hardware from tlp_median_14_time mean = " + str(np.mean(diff_median_14)) + "  var = " + str(np.var(diff_median_14)) + '\n')
    #     # Write each item in the list to a new line in the file
    #     file.write("diff_home_14: ")
    #     for item in diff_home_14:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("diff_median_14: ")
    #     for item in diff_median_14:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("diff_home0_13: ")
    #     for item in diff_home0_13:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("diff_old_14: ")

    #     for item in diff_old_14:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("hardware_time: ")
    #     for item in hardware_time:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("tlp_median_home_14_time: ")
    #     for item in tlp_median_home_14_time:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("tlp_median_14_time: ")
    #     for item in tlp_median_14_time:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("tlp_median_home0_13_time: ")
    #     for item in tlp_median_home0_13_time:
    #         file.write(str(item) + '  ')
    #     file.write('\n')
    #     file.write("tlp_old_14_time: ")
    #     for item in tlp_old_14_time:
    #         file.write(str(item) + '  ')


# To make a GFlowNet dataset
# TODO: need for add workload(in context) info as condition
def gflownet_data_save(data_path, save_path, database_path, decision_path):
    assert os.path.exists(data_path), f"{data_path} not exists!"
    # database include candidates(trace, instructions&decision) & workload(subgraph)
    databases = load_all_files(data_path)
    gm = GflowNetEmbedding()
    print("Successfully Load Databases!")
    datasets = []
    count_ptr = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(database_path):
        os.makedirs(database_path)

    if not os.path.exists(decision_path):
        os.makedirs(decision_path)
    # get min cost time from multiple measure runtime

    def _min_cost(res) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.min([float(s) for s in res.run_secs]))

    max_order_len = 0
    file_name = []

    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        cc = 0
        for record in records:

            # convert record into measured candidates
            # measure_candidate for
            sub_sch = record.as_measure_candidate().sch
            # record.workload is workload info
            min_cost = _min_cost(record)
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions

            from tvm.ir.container import Array
            from tvm.tir.expr import IntImm

            def custom_sort(item):
                value = item[1]
                if isinstance(value, Array):
                    val = [int(v.value) for v in value]
                    return sum(val)  # 对列表值求和作为排序关键字
                else:
                    return int(value)
            sub_decisions = sorted(
                dict(sub_decisions).items(), key=custom_sort)
            sub_decisions = {k: v for k, v in sub_decisions}

            candidate = record.as_measure_candidate()
            # print(f"old insts & decision = {sub_decisions}")
            # print(f"old insts = {list(sub_decisions.keys())}")
            print(f"old decision = {list(sub_decisions.values())}")

            extend_embedding_0 = []
            extend_embedding_1 = []
            ex_cond0 = []
            ex_cond1 = []
            # list(3, 10) (3, 24) (3, 1) -- anno/cuda
            # (4, 32, 10) (4, 34) (3, 3, 7) -- sample tile
            # NOTE: save new order decision and insts!!!
            embedding_results, embedding_conditions, count_ptr_list, new_decisions = gm(
                sub_insts, sub_decisions, True)

            # print(f"new insts = {list(new_decisions.keys())}")
            print(f"new decision = {list(new_decisions.values())}")
            import json

            # 保存字典到文件
            def save_dict_to_file(dictionary, filename):
                with open(filename, 'w') as file:
                    json.dump(dictionary, file)

            # 从文件中读取字典
            def load_dict_from_file(filename):
                with open(filename, 'r') as file:
                    dictionary = json.load(file)
                return dictionary

            # # 保存new order decision到文件
            # save_dict_to_file(new_decisions, os.path.join(
            #         decision_path, f"decisions_{count_ptr}.json", ),)

            # # 从文件中读取字典
            # loaded_dict = load_dict_from_file('data.json')

            # NOTE: new decision is null list -- gm must pass valid insts & decisions
            # print(f"new decision = {new_sub_decisions}")

            # # Must use with_decision() to set sub_trace
            # for new_sub_inst, new_sub_decision in new_decisions.items():
            #     # new_sub_decision = tvm.tir.const(1, dtype='int32')
            #     # NOTE: bug1 must assign to sub_trace
            #     sub_trace = sub_trace.with_decision(
            #         new_sub_inst, new_sub_decision, True)

            # print(f"trace insts = {list(sub_trace.decisions.keys())}")
            print(f"trace decision = {list(sub_trace.decisions.values())}")
            from tvm.meta_schedule.database.database import TuningRecord
            target = "cuda"

            new_database = ms.database.JSONDatabase(
                path_workload=os.path.join(
                    database_path, f"workloads_{count_ptr}.json", ),
                path_tuning_record=os.path.join(
                    database_path, f"candidates_{count_ptr}.json"),
            )
            workload = record.workload
            new_database.commit_workload(workload.mod)
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=sub_trace,
                    workload=workload,
                    run_secs=record.run_secs,
                    target=Target(target),
                    args_info=candidate.args_info
                )
            )
            print(f"Successfully Save File database_{count_ptr}")

            decode = []
            # NOTE: (cond_x1, cond_y1) is anno&cuda shape (cond_x2, cond_y2) is tile shape
            # cond_x = len(embedding_conditions)
            # cond_y = embedding_conditions[0].shape[0]
            # decode += (cond_x, cond_y)
            cond_x1 = 0
            cond_y1 = 0
            cond_x2 = 0
            cond_y2 = 0
            order = []
            max_len = 0
            for ii in range(len(embedding_results)):
                embedding = embedding_results[ii]
                cond = embedding_conditions[ii]
                max_len += 1
                _len = embedding.shape[0]
                if _len > 10:  # If the primitive type is the sample perfectile -- len = 32
                    order.append(1)
                    cond_x2 += 1
                    cond_y2 = embedding_conditions[max_len-1].shape[0]  # 34
                    extend_embedding_1.append(
                        torch.from_numpy(embedding))  # append (32, 10)
                    ex_cond1.append(torch.from_numpy(
                        cond.squeeze().astype(float)))
                    # print("Sample Perfect Tile shape: ", extend_embedding_1[-1].shape)
                else:  # prim type is other type: annotation & cuda bind -- len = 10
                    order.append(0)
                    cond_x1 += 1
                    cond_y1 = embedding_conditions[max_len-1].shape[0]  # 24
                    extend_embedding_0.append(
                        torch.from_numpy(embedding.squeeze()))
                    ex_cond0.append(torch.from_numpy(
                        cond.astype(float).squeeze()))
                    # print("Annotation & CUDA Bind shape: ", extend_embedding_0[-1].shape)
            if max_len > max_order_len:
                max_order_len = max_len

            decode += [cond_x1, cond_y1, cond_x2, cond_y2, max_len]
            # NOTE: Padding to max length for embeddings
            # TODO: Need padding condition
            MAX_N = 15
            # NOTE: padding order into MAX_N
            while len(order) < MAX_N:
                order.append(-1)

            # stack for convert [(10, ), (10, )..] into (3, 10)
            if len(extend_embedding_0) > 0:
                extend_embedding_0 = torch.stack(extend_embedding_0, 0)
                ex_cond0 = torch.stack(ex_cond0, 0)
            else:  # first shape is 15*10 -- binary vector
                extend_embedding_0 = torch.zeros(MAX_N, 10)
                ex_cond0 = torch.zeros(MAX_N, 24)
            # stack for convert [(32, 10, ), (32, 10, )..] into (6, 32, 10)
            if len(extend_embedding_1) > 0:
                extend_embedding_1 = torch.stack(extend_embedding_1, 0)
                ex_cond1 = torch.stack(ex_cond1, 0)
            else:  # second shape is 15*320 -- binary vector

                extend_embedding_1 = torch.zeros(MAX_N, 32, 10)
                ex_cond1 = torch.zeros(MAX_N, 34)

            # NOTE: add embedding 0/1 shape into decode info
            sz1 = extend_embedding_0.shape[0]
            sz2 = extend_embedding_1.shape[0]
            decode += (sz1, sz2)
            m0, m1, m2 = extend_embedding_1.shape
            # NOTE: padding zeros, shape[1] is same -- convert into (15, ..)
            extend_embedding_0 = torch.cat([extend_embedding_0,
                                            torch.zeros(MAX_N - sz1, extend_embedding_0.shape[1]).to(extend_embedding_0.device)], 0)
            extend_embedding_1 = torch.cat([extend_embedding_1,
                                            torch.zeros(MAX_N - sz2, m1, m2).to(extend_embedding_1.device)], 0)

            # NOTE: padding zeros, shape[1] is same -- convert into (15, ..)
            ex_cond0 = torch.cat([ex_cond0,
                                  torch.zeros(MAX_N - sz1, ex_cond0.shape[1]).to(ex_cond0.device)], 0)
            ex_cond1 = torch.cat([ex_cond1,
                                  torch.zeros(MAX_N - sz2, ex_cond1.shape[1]).to(ex_cond1.device)], 0)

            # Now extend_embedding_0's shape is (15,10), and extend_embedding_1's shape is (15,320)
            # After that, we flatten and concatenate them.
            # Translate one-hot (15, 10) to label (15, ), one-hot (15, 32, 10) to (15, 32)
            extend_embedding_0 = torch.argmax(extend_embedding_0, -1)
            extend_embedding_1 = torch.argmax(extend_embedding_1, -1)
            extend_embedding_1 = extend_embedding_1.flatten()  # Flatten it into (15*32)

            # Concatenate them, the last_embedding's shape is (15+15*32, ) = (495)
            last_embedding = torch.cat(
                [extend_embedding_0, extend_embedding_1], 0)
            # NOTE: padding cond0 into 34 = ex_cond1.shape[1]
            ex_cond0 = torch.cat([ex_cond0,
                                  torch.zeros(MAX_N, 10).to(ex_cond0.device)], 1)
            last_condition = torch.cat([ex_cond0, ex_cond1], 0)

            last_ptr_list = torch.Tensor(count_ptr_list)

            # print("last embedding shape: ", last_embedding.shape) # torch.Size([1455])
            # print("not padding condition shape: ", last_condition.shape) # torch.Size([72])
            # print("last ptr list shape: ", last_ptr_list.shape) # torch.Size([3])
            # NOTE: We define attr in dataset: last_embedding, last_condition, last_ptr_lis, run_secs
            np.savez(os.path.join(save_path, f'mlc_{count_ptr}.npz'), decode=decode,
                     order=order, last_embedding=last_embedding, last_condition=last_condition,
                     last_ptr_list=last_ptr_list, run_secs=min_cost)
            print(f"Successfully Save File mlc_{count_ptr}.npz")
            count_ptr += 1
            file_name.append(str(count_ptr)+"  " +
                             database.path_workload+f"  {cc}")
            cc += 1

    with open(os.path.join(database_path, f'0records.txt'), 'w') as file:
        # Write each item in the list to a new line in the file
        for item in file_name:
            file.write(str(item) + '\n')

    with open(os.path.join(save_path, f'0records.txt'), 'w') as file:
        # Write each item in the list to a new line in the file
        for item in file_name:
            file.write(str(item) + '\n')


def tlp_data_save(data_path, save_path):
    target = "nvidia/nvidia-a100"
    count_ptr = 0

    databases = load_all_files(data_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Successfully Load Databases!")
    candidates, results = [], []
    file_name = []
    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        cc = 0
        for record in records:
            candidates.append(record.as_measure_candidate())
            results.append(RunnerResult(
                run_secs=record.run_secs, error_msg=None))
            context = TuneContext(mod=record.workload.mod,
                                  target=Target(target))

            new_database = ms.database.JSONDatabase(
                path_workload=os.path.join(
                    save_path, f"workloads_{count_ptr}.json", ),
                path_tuning_record=os.path.join(
                    save_path, f"candidates_{count_ptr}.json"),
            )
            workload = record.workload
            new_database.commit_workload(workload.mod)
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    run_secs=record.run_secs,
                    target=Target(target),
                )
            )
            print(f"Successfully Save File database_{count_ptr}")
            count_ptr += 1
            file_name.append(database.path_workload+f"{cc}")
            cc += 1

    with open(os.path.join(save_path, f'0records.txt'), 'w') as file:
        # Write each item in the list to a new line in the file
        for item in file_name:
            file.write(str(item) + '\n')


def worker1(workload_path):

    candidate_path = workload_path.replace("workloads", "candidates")

    database = ms.database.JSONDatabase(
        path_workload=workload_path, path_tuning_record=candidate_path)
    # print("In each worker, ", len(database.get_all_tuning_records()))

    return database.get_all_tuning_records()


def worker0(workload_path):

    candidate_path = workload_path.replace("workloads", "candidates")

    database = ms.database.JSONDatabase(
        path_workload=workload_path, path_tuning_record=candidate_path)
    # print("In each worker, ", len(database.get_all_tuning_records()))
    return database


def record_data_load(record_path):

    pool = multiprocessing.Pool(112)
    # pool = ThreadPool()
    workload_paths = glob.glob(os.path.join(
        record_path, f"*workloads_*.json"), recursive=True)

    databases = pool.map(worker1, workload_paths)

    print(len(databases))
    for i in range(20):
        records = databases[i]
        print(len(records))

    return databases


def record_data_load0(record_path):

    from joblib import Parallel, delayed

    # Parallelize the for loop using Joblib

    workload_paths = glob.glob(os.path.join(
        record_path, f"*workloads_*.json"), recursive=True)

    databases = Parallel(n_jobs=112)(delayed(worker0)(i)
                                     for i in workload_paths)
    print("Finish load!")

    print(len(databases))
    for i in range(20):
        records = databases[i].get_all_tuning_records()
        print("len of records = ", len(records))

    return databases


def extract_features(
    context: TuneContext,
    candidates: List[MeasureCandidate],
    results: Optional[List[RunnerResult]] = None,
    extractor: Optional[FeatureExtractor] = None,
):

    extractor = extractor or PerStoreFeature(extract_workload=True)

    def _feature(feature: NDArray) -> np.ndarray:
        return feature.numpy().astype("float32")

    def _mean_cost(res: RunnerResult) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.median([float(s) for s in res.run_secs]))

    new_features = [_feature(x)
                    for x in extractor.extract_from(context, candidates)]
    #  np.array([_mean_cost(x) for x in results]).astype("float32")
    new_mean_costs = None
    new_mean_costs = (
        np.array([_mean_cost(x) for x in results]).astype("float32")
        if results is not None
        else None
    )

    return new_features, new_mean_costs


def restore_embedding(decode_info):

    MAX_N = 15
    ROW = 32
    COL = 10
    # (bs, ...)
    xs, databases_path, decodes, orders, conds, ptrs, target = decode_info
    bs = xs.shape[0]
    contexts, candidates = [], []
    # print("len of xs", len(xs))
    # TODO:.. print(len)
    old_decision = []
    new_decision = []

    for i in range(bs):
        x, database_path, decode, order, cond, ptr = \
            xs[i], databases_path[i], decodes[i], orders[i], conds[i], ptrs[i]

        cond_x0, cond_y0, cond_x1, cond_y1, max_len, emb0_x, emb1_x = decode
        ex_emb0, ex_emb1 = torch.split(x, [MAX_N, MAX_N*ROW], 0)
        ex_cond0, ex_cond1 = torch.split(
            cond, split_size_or_sections=MAX_N, dim=0)

        ex_emb1 = ex_emb1.view(MAX_N, -1)
        # # convert into one-hot format
        # ex_emb0 = torch.eye(10)[ex_emb0.to(torch.int64)]
        # ex_emb1 = torch.eye(10)[ex_emb1.to(torch.int64)]

        emb0_x = emb0_x.item()
        emb1_x = emb1_x.item()

        emb0, _ = torch.split(ex_emb0, [emb0_x, MAX_N-emb0_x], 0)
        emb1, _ = torch.split(ex_emb1, [emb1_x, MAX_N-emb1_x], 0)

        cond0, _ = torch.split(ex_cond0, [cond_x0, MAX_N-cond_x0], 0)
        cond1, _ = torch.split(ex_cond1, [cond_x1, MAX_N-cond_x1], 0)

        cond0, _ = torch.split(cond0, [cond_y0, 34-cond_y0], 1)
        cond1, _ = torch.split(cond1, [cond_y1, 34-cond_y1], 1)
        # convert cuda:0 device into cpu
        emb0 = emb0.cpu()
        emb1 = emb1.cpu()
        cond0 = cond0.cpu()
        cond1 = cond1.cpu()
        # print(f"emb0 = {emb0}")
        # print(f"emb1 = {emb1}")
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

        # candidate = record.as_measure_candidate()
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

        from tvm.ir.container import Array
        from tvm.tir.expr import IntImm

        def custom_sort(item):
            value = item[1]
            if isinstance(value, Array):
                val = [int(v.value) for v in value]
                return sum(val)  # 对列表值求和作为排序关键字
            else:
                return int(value)
        # NOTE: MUST sorted for fixed position!!!
        sub_decisions = sorted(
            dict(sub_decisions).items(), key=custom_sort)
        sub_decisions = {k: v for k, v in sub_decisions}

        # print(f"old decision = {list(sub_decisions.values())}")
        # print(f"res = {res}")

        gm = GflowNetEmbedding()
        new_sub_decisions = gm(sub_insts, sub_decisions, False, embedding_results=res,
                               embedding_conditions=emb_conds, count_Ptr_results=ptr)

        # NOTE: new decision is null list -- gm must pass valid insts & decisions
        # print(f"new decision = {new_sub_decisions}")

        # Must use with_decision() to set sub_trace
        for new_sub_inst, new_sub_decision in new_sub_decisions.items():
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

        # if candidate_path == "/root/share/dataset/tlp_dataset0/candidates_64.json":
        #     print(f"Wrong decision, json = {candidate_path}")
        #     print(f"Wrong decision, ints * decision = {new_sub_decisions}")

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

        # raise ValueError(f"{e}")

        context = TuneContext(mod=record.workload.mod, target=Target(target))
        # TODO: check same context
        # if len(contexts) > 0:
        #     if context != contexts[-1]:
        #         print("diff context")

        # check correctness for
        # print(f"final candidate decision = {list(sub_decisions.values())}")
        contexts.append(context)
        candidates.append(candidate)
        # print(f"after record = {record}, candidate = {candidate}, decision = ")

    # print(f"old decision = {old_decision}")
    # print(f"new decisions = {new_decision}")
    features, _ = extract_features(contexts[0], candidates)

    return features


def formatter(file, workload_paths, candidate_paths):
    decode = file["decode"]
    decode = torch.from_numpy(decode)
    order = file["order"]
    order = torch.from_numpy(order)
    last_embedding = file['last_embedding']
    last_embedding = torch.from_numpy(last_embedding)
    last_condition = file['last_condition']
    last_condition = torch.from_numpy(last_condition)

    last_ptr_list = file['last_ptr_list']
    last_ptr_list = torch.from_numpy(last_ptr_list)
    run_secs = file['run_secs']
    run_secs = torch.from_numpy(run_secs)

    return last_embedding, workload_paths, candidate_paths, decode, order, last_condition, last_ptr_list, run_secs

# dataset for GFlowNet: [15+15*32] 15 is 0~9 for anno+cuda
# GFlowNet prefer 0~9 format avoid invalid format [1, 0, 1] + extra mask -- GFlowNet output [0.3, 0.5, ..., 0.1] with 10 position
# 15*32 is 0~9 for tile (质因数分解) allowing [0, 1, 1] not one-hot format -- GFlowNet output [[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]]


class MLCGflowNetDataset(Dataset):
    # TODO: change without_condition=False & determine condition len in future
    def __init__(self, all_files,):
        self.npz_files, self.workload_paths, self.candidate_paths = all_files

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        file = self.npz_files[idx]
        file = np.load(file)
        workload_paths, candidate_paths = self.workload_paths[idx], self.candidate_paths[idx]
        return formatter(file, workload_paths, candidate_paths)

# # Custom collate function

# def my_collate_fn(batch):
#     strings = []
#     numbers = []

#     for item in batch:
#         if isinstance(item, str):
#             strings.append(item)
#         else:
#             numbers.append(item)

#     return strings, torch.tensor(numbers)

# Load above GFlowNet Dataset for search, ret dataloader


def gflownet_data_load(save_path,
                       database_path=None,
                       num_workers=112,
                       batch_size=16,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=True):
    import glob

    def search_all_files(work_dir, database_path):
        npz_files = glob.glob(os.path.join(
            work_dir, "*.npz"), recursive=True)

        workload_paths = glob.glob(os.path.join(
            database_path, f"workloads*.json"), recursive=True)
        candidate_paths = [workload_path.replace(
            "workloads", "candidates") for workload_path in workload_paths]

        # num = len(candidate_paths)
        # # NOTE: not use sorted()
        # databases_path = [(workload_paths[i], candidate_paths[i])
        #                   for i in range(num)]
        return npz_files, workload_paths, candidate_paths

    all_files = search_all_files(save_path, database_path)
    dataset = MLCGflowNetDataset(all_files)
    # A generative task without any validate dataset.
    # DataLoader speed up data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return dataloader, len(dataset)
