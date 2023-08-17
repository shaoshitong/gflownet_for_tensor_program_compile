import argparse
import os
import json
import glob
import torch
from collections import OrderedDict
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

import tvm
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule import TuneContext, FeatureExtractor,MeasureCandidate
from tvm.target import Target
from tvm.meta_schedule.feature_extractor import PerStoreFeature
from tvm.runtime import NDArray
from tvm.meta_schedule.utils import  shash2hex
import multiprocessing 


# pylint: disable=too-few-public-methods
class FeatureGroup:
    """Feature group

    Parameters
    ----------
    group_hash : str
        The hash of the group
    features : List[np.ndarray]
        The features
    costs : List[float]
        The costs
    min_cost : float
        The minimum cost
    """

    group_hash: str
    features: List[np.ndarray]
    costs: np.ndarray
    min_cost: float

    def __init__(
        self,
        group_hash: str,
        features: List[np.ndarray],
        costs: np.ndarray,
        min_costs = None
    ) -> None:
        self.group_hash = group_hash
        self.features = features
        self.costs = costs
        self.min_cost = np.min(costs) if min_costs==None else min_costs

    def append(  # pylint: disable=missing-function-docstring
        self,
        features: List[np.ndarray],
        costs: np.ndarray,
    ) -> None:
        self.features.extend(features)
        self.costs = np.append(self.costs, costs)
        self.min_cost = np.min(self.costs)

    def to_json(self):
        return {"features": [feature.tolist() for feature in self.features], "mean_costs":self.costs.tolist(), "min_cost":float(self.min_cost) } 
    
    @classmethod
    def from_json(cls, json_obj):
        key = list(json_obj.keys())[0]
        data = list(json_obj.values())[0]
        return cls(key, data["features"], data["mean_costs"], data["min_cost"])        



def extract_features(
    context: TuneContext,
    candidates: List[MeasureCandidate],
    results: Optional[List[RunnerResult]] = None,
    extractor: Optional[FeatureExtractor] = None,
):
    """Extract feature vectors and compute mean costs.

    Parameters
    ----------
    context: TuneContext
        The tuning context.
    candidates: List[MeasureCandidate]
        The measure candidates.
    results: Optional[List[RunnerResult]]
        The measured results, can be None if used in prediction.
    extractor: Optional[FeatureExtractor]
        The feature extractor.

    Returns
    -------
    new_features: List[np.ndarray]
        The extracted features.
    new_mean_costs: np.ndarray
        The mean costs.
    """
    extractor = extractor or PerStoreFeature(extract_workload=True)

    def _feature(feature: NDArray) -> np.ndarray:
        return feature.numpy().astype("float32")

    def _mean_cost(res: RunnerResult) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.median([float(s) for s in res.run_secs]))

    new_features = [_feature(x) for x in extractor.extract_from(context, candidates)]
    new_mean_costs = (
        np.array([_mean_cost(x) for x in results]).astype("float32")
        if results is not None
        else None
    )
    return new_features, new_mean_costs


def add_to_group(
    data: Dict[str, FeatureGroup],
    features: List[np.ndarray],
    costs: np.ndarray,
    group_hash: str,
):
    group = data.get(group_hash, None)
    if group == None:
        group = FeatureGroup(
            group_hash=group_hash,
            features=features,
            costs=costs
        )
    else:
        group.append(features, costs)
    data[group_hash] = group
    return data

def handle_json(model_dir:str):
    workload_paths = []
    candidate_path = []
    json_files = glob.glob(os.path.join(model_dir, "*.json"))
    for json_file in json_files:
        if json_file.endswith("_workload.json"):
            workload_paths.append(json_file)
        elif json_file.endswith("_candidates.json"):
            candidate_path.append(json_file)
    #handhle json file
    extractor_feature = PerStoreFeature(extract_workload=True)
    for workload_path in tqdm(workload_paths):
        try:
            database = JSONDatabase(
                path_workload=workload_path,
                path_tuning_record=workload_path.replace(
                    "_workload.json", "_candidates.json"
                ),
            )
        except tvm._ffi.base.TVMError:  # pylint: disable=protected-access
            continue
        all_dataset = OrderedDict()
        candidates, results = [], []
        tuning_records = database.get_all_tuning_records()
        if len(tuning_records) == 0:
            continue
        for record in tuning_records:
            candidates.append(record.as_measure_candidate())
            results.append(RunnerResult(run_secs=record.run_secs, error_msg=None))
        assert len(candidates) == len(results)
        context = TuneContext(mod=tuning_records[0].workload.mod, target=Target(args.target))
        features, mean_costs = extract_features(
            context, candidates, results, extractor_feature
        )
        all_dataset = add_to_group(all_dataset, features, mean_costs, shash2hex(context.mod))
        #save feature json file  
        model_name,_ = os.path.splitext(os.path.basename(model_dir))
        file_name, _ =os.path.splitext(os.path.basename(workload_path))
        file_name = file_name.rsplit('_',1)[0]
        need_saved = {k:v.to_json() for k, v in all_dataset.items()}
        with open(f'{args.save_folder}/{model_name}_{file_name}_train_and_val.json', 'w') as f:
            json.dump(need_saved, f)
    

def make_all_dataset():
    os.makedirs(args.save_folder, exist_ok=True)
    
    pool = multiprocessing.Pool()
    model_dirs = glob.glob(os.path.join(args.dataset_path, "*"))
    for model_dir in model_dirs:
        pool.apply_async(handle_json, args=(model_dir,))
    pool.close()
    pool.join()
            


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_path",type=str, default= './dataset_a100/measure_candidate')
    parse.add_argument("--save_folder", type=str, default='')
    parse.add_argument("--target",type=str, default='cuda')

    args = parse.parse_args()
    
    if args.save_folder == '':
        args.save_folder = os.path.join(os.path.dirname(args.dataset_path), 'extract_features')

    make_all_dataset()    