import os
import sys
import numpy as np
from src.gfn.mlc_dataset import *
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
        np.savez(os.path.join(info_path, f'info{step}.npz'), decode=decode,
                    order=order, last_embedding=x, last_condition=cond,
                    last_ptr_list=ptr, run_secs=score)
