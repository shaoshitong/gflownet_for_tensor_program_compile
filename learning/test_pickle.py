# import torch

# print("Torch version:",torch.__version__)

# print("Is CUDA enabled?",torch.cuda.is_available())

import pickle
from tvm.meta_schedule.cost_model.tlp_cost_model_train import TransformerModule
path = "/root/kongdehao/model/tlp_model_14.pkl"

with open(path, 'rb') as f:
    model = pickle.load(f)

# moduleList = pickle.load(open('/home/user/mysite/modules.p','rb'))
# moduleList = pickle.load(open('/home/user/mysite/modules.p','rb'))
model.to("cuda")

model.eval()