import os
import torch
from typing import List

import numpy as np
import tvm
import tvm.testing
from tvm.runtime import NDArray
from tvm.meta_schedule.cost_model import PyCostModel, RandomModel, XGBModel
from tvm.meta_schedule.cost_model.xgb_model import PackSum, _get_custom_call_back
from tvm.meta_schedule.feature_extractor import FeatureExtractor,PerStoreFeature
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.tir.schedule.schedule import Schedule
import tvm.meta_schedule as ms
from .tlp_cost_model_train import *


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



@derived_object
class tlpCostModel(PyCostModel):
    
    def __init__(self, *, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.loss_func = LambdaRankLoss(self.device)
    def load(self, path: str = "./tlp_model_14.pkl") -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)  
        self.model.to(self.device)
          
    def save(self, path: str) -> None:
        pass
    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        pass
    
    def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
        self.model.eval()
        features, _ = extract_features(context, candidates)
        val_dataloader = SegmentDataLoader(
            features,shuffle=False
        )
        pred_results = []
        for batch_data,_ in val_dataloader:
            batch_data = batch_data.to(self.device)
            outputs = self.model(batch_data)
            pred_results.extend(outputs.detach().cpu().numpy())
        return pred_results