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


def tlp_data_save(data_path, save_path):
    target = "nvidia/nvidia-a100"
    count_ptr = 0

    databases = load_all_files(data_path)
    print("Successfully Load Databases!")
    candidates, results = [], []
    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
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
    workload_paths = sorted(
        glob.glob(os.path.join(record_path, f"*workloads_*.json"), recursive=True))

    databases = pool.map(worker1, workload_paths)

    print(len(databases))
    for i in range(20):
        records = databases[i]
        print(len(records))

    return databases


def record_data_load0(record_path):

    from joblib import Parallel, delayed

    # Parallelize the for loop using Joblib

    workload_paths = sorted(
        glob.glob(os.path.join(record_path, f"*workloads_*.json"), recursive=True))

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

    import torch.nn.functional as Fun
    MAX_NUMBER = 15
    # (bs, ...)
    xs, databases_path, decodes, orders, conds, ptrs, target = decode_info
    bs = xs.shape[0]
    contexts, candidates = [], []
    # print("len of xs", len(xs))
    # TODO:.. print(len)

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
        # print(f"old decision = {list(sub_decisions.values())}")
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

        print(f"After decision = {list(sub_trace.decisions.values())}")
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
        candidate = record.as_measure_candidate()
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

    features, _ = extract_features(contexts[0], candidates)

    return features


# To make a GFlowNet dataset
# TODO: need for add workload(in context) info as condition
def gflownet_data_save(data_path, save_path):
    assert os.path.exists(data_path), f"{data_path} not exists!"
    # database include candidates(trace, instructions&decision) & workload(subgraph)
    databases = load_all_files(data_path)
    gm = GflowNetEmbedding()
    print("Successfully Load Databases!")
    datasets = []
    count_ptr = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get min cost time from multiple measure runtime
    def _min_cost(res) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.min([float(s) for s in res.run_secs]))

    max_order_len = 0

    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        for record in records:
            if os.path.exists(f"mlc_{count_ptr}.npz"):
                count_ptr += 1
                print(f"Passing mlc_{count_ptr}.npz")
                continue
            # convert record into measured candidates
            # measure_candidate for
            sub_sch = record.as_measure_candidate().sch
            # record.workload is workload info
            min_cost = _min_cost(record)
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions

            extend_embedding_0 = []
            extend_embedding_1 = []
            ex_cond0 = []
            ex_cond1 = []
            # list(3, 10) (3, 24) (3, 1) -- anno/cuda
            # (4, 32, 3) (4, 34) (3, 3, 7) -- sample tile
            embedding_results, embedding_conditions, count_ptr_list = gm(
                sub_insts, sub_decisions, True)

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
                        torch.from_numpy(embedding.reshape(-1)))  # reshape (32, 3) -- (96)
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
            MAX_NUMBER = 15
            # NOTE: padding order into MAX_NUMBER
            while len(order) < MAX_NUMBER:
                order.append(-1)

            # stack for convert [(10, ), (10, )..] into (3, 10)
            if len(extend_embedding_0) > 0:
                extend_embedding_0 = torch.stack(extend_embedding_0, 0)
                ex_cond0 = torch.stack(ex_cond0, 0)
            else:  # first shape is 15*10 -- binary vector
                extend_embedding_0 = torch.zeros(MAX_NUMBER, 10)
                ex_cond0 = torch.zeros(MAX_NUMBER, 24)
            # stack for convert [(96, ), (96, )..] into (6, 96)
            if len(extend_embedding_1) > 0:
                extend_embedding_1 = torch.stack(extend_embedding_1, 0)
                ex_cond1 = torch.stack(ex_cond1, 0)
            else:  # second shape is 15*96 -- binary vector

                extend_embedding_1 = torch.zeros(MAX_NUMBER, 96)
                ex_cond1 = torch.zeros(MAX_NUMBER, 34)

            # NOTE: add embedding 0/1 shape into decode info
            sz1 = extend_embedding_0.shape[0]
            sz2 = extend_embedding_1.shape[0]
            decode += (sz1, sz2)

            # NOTE: padding zeros, shape[1] is same -- convert into (15, ..)
            extend_embedding_0 = torch.cat([extend_embedding_0,
                                            torch.zeros(MAX_NUMBER - sz1, extend_embedding_0.shape[1]).to(extend_embedding_0.device)], 0)
            extend_embedding_1 = torch.cat([extend_embedding_1,
                                            torch.zeros(MAX_NUMBER - sz2, extend_embedding_1.shape[1]).to(extend_embedding_1.device)], 0)

            # NOTE: padding zeros, shape[1] is same -- convert into (15, ..)
            ex_cond0 = torch.cat([ex_cond0,
                                  torch.zeros(MAX_NUMBER - sz1, ex_cond0.shape[1]).to(ex_cond0.device)], 0)
            ex_cond1 = torch.cat([ex_cond1,
                                  torch.zeros(MAX_NUMBER - sz2, ex_cond1.shape[1]).to(ex_cond1.device)], 0)

            # Now extend_embedding_0's shape is (15,10), and extend_embedding_1's shape is (15,96)
            # After that, we flatten and concatenate them.
            # Translate one-hot to label (15, )
            extend_embedding_0 = torch.argmax(extend_embedding_0, -1)
            extend_embedding_1 = extend_embedding_1.flatten()  # Flatten it into (1440, )

            # Concatenate them, the last_embedding's shape is (15+15*96, ) = (1455)
            last_embedding = torch.cat(
                [extend_embedding_0, extend_embedding_1], 0)
            # NOTE: padding cond0 into 34 = ex_cond1.shape[1]
            ex_cond0 = torch.cat([ex_cond0,
                                  torch.zeros(MAX_NUMBER, 10).to(ex_cond0.device)], 1)
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

    print("Max order len = ", max_order_len)


def formatter(file):
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

    return decode, order, last_embedding, run_secs, last_condition, last_ptr_list

# dataset for GFlowNet: [15+15*96] 15 is 0~9 for anno+cuda
# GFlowNet prefer 0~9 format avoid invalid format [1, 0, 1] + extra mask -- GFlowNet output [0.3, 0.5, ..., 0.1] with 10 position
# 15*96 is 0~1 for tile (质因数分解) allowing [0, 1, 1] not one-hot format -- GFlowNet output [[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]]


class MLCGflowNetDataset(Dataset):
    # TODO: change without_condition=False & determine condition len in future
    def __init__(self, all_files, without_condition=False, ):
        self.all_files = all_files
        self.without_condition = without_condition

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file = self.all_files[idx]
        file = np.load(file)
        return formatter(file)


# Load above GFlowNet Dataset for search, ret dataloader
def gflownet_data_load(save_path,
                       without_condition=False,
                       num_workers=4,
                       batch_size=16,
                       shuffle=False,
                       pin_memory=False,
                       drop_last=True):
    import glob

    def search_all_files(work_dir):
        npz_files = sorted(glob.glob(os.path.join(
            work_dir, "*.npz"), recursive=True))
        return npz_files

    all_files = search_all_files(save_path)
    dataset = MLCGflowNetDataset(
        all_files, without_condition=without_condition)
    # A generative task without any validate dataset.
    # DataLoader speed up data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return dataloader, len(dataset)
