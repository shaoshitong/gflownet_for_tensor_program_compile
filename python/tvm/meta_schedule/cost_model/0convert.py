import torch  # type: ignore
from torch import nn
import pickle
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

device = "cuda"
path = "/root/share/dataset/tlp/save_model_v1/tlp_model_14.pkl"
with open(path, 'rb') as f:
    model = pickle.load(f)
model.to(device)
model = model.module
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
save_path = "/root/share/dataset/tlp/save_model_v1/tlp_14.pth"
torch.save(model, save_path)

print("Finish")
