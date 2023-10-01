import os
import torch
from typing import List

import numpy as np

from ...contrib.tar import tar, untar
from ...runtime import NDArray
from ..cost_model import PyCostModel
from ..feature_extractor import FeatureExtractor, PerStoreFeature
from ..logging import get_logger
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..utils import cpu_count, derived_object, shash2hex
from .metric import max_curve
from ..tune_context import TuneContext
from typing import Dict, List, NamedTuple, Optional, Tuple
from .tlp_cost_model_train import *
# from .tlp_cost_model_train import TransformerModule

# from tvm.meta_schedule.cost_model.tlp_cost_model_train import TransformerModule


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
    # NOTE: cuda:7 is corresponding to 
    def __init__(self, *, device='cuda:0') -> None:
        super().__init__()
        # print("----------------------------Enter __init__ func")
        self.device = device
        # TODO: 数据集中 是否存在class的信息，否则需要进行修改！
        self.loss_func = LambdaRankLoss(self.device)
        # NOTE: For load() must call in __init__().
        self.load()
        # NOTE: cann't use "./tlp_model_14.pkl", cann't find file or dir
        # NOTE: update relative path to tlp model
    # def load(self, path: str = "python/tvm/meta_schedule/cost_model/tlp_model_14.pkl") -> None:
    def load(self, path: str = "/root/kongdehao/model/tlp/tlp_model_73.pth") -> None:
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&Enter load func")
        # self.model = TransformerModule().to(self.device)
        self.model = torch.load(path, map_location=self.device)
        # self.model.load_state_dict(checkpoint["tlp"])
        
        # if not os.path.exists(path):
        #     path = "../" + path
        #     if not os.path.exists(path):
        #         raise NotImplementedError(f"{path} not exists!")

        # with open(path, 'rb') as f:
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$Open tlp model")
        #     name, self.model = torch.load(f)  
        # self.model.to(self.device)
          
    def save(self, path: str) -> None:
        pass
    
    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        pass
    
    @torch.no_grad()
    def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
        self.model.eval()
        features, _ = extract_features(context, candidates)
        # NOTE: This is for speed up predict!
        val_dataloader = SegmentDataloder_new(
            features, shuffle=False
        )
        pred_results = []
        for batch_data,_ in val_dataloader:
            batch_data = batch_data.to(self.device)
            outputs = self.model(batch_data)
            pred_results.extend(outputs.detach().cpu().numpy())
        return pred_results