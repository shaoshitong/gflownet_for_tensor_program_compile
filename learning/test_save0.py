import torch
import torch.nn as nn
# NOTE: must to import nn module
from test_save import MyModule

dev = "cuda"
load_model = torch.load("model.pth",map_location=dev)

print("Load Model = ", load_model)