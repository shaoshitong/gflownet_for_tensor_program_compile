import tvm,os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tvm.meta_schedule.cost_model.tlp_cost_model_train import from_json,load_data
# from .dataset_embedding import GflowNetEmbedding, load_all_files

# mymodule.py
import os
import sys
# Add the parent directory of mypackage to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import subpackage.submodule
# from mypackage.subpackage import submodule
from .dataset_embedding import GflowNetEmbedding, load_all_files


# To make a GFlowNet dataset
# TODO: need for add workload(in context) info as condition
def gflownet_data_save(data_path,save_path):
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
    
    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        for record in records:
            if os.path.exists(f"mlc_{count_ptr}.npz"):
                count_ptr+=1
                print(f"Passing mlc_{count_ptr}.npz")
                continue
            # convert record into measured candidates
            sub_sch = record.as_measure_candidate().sch
            # record.workload is workload info
            min_cost = _min_cost(record)
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions
            
            extend_embedding_0 = []
            extend_embedding_1 = []
            embedding_results,embedding_conditions,count_ptr_list = gm(sub_insts,sub_decisions,True)


            for cond in embedding_conditions:
                print("Condition shape: ", cond.shape)

            for embedding in embedding_results:
                _len = embedding.shape[0]
                if _len > 10: # If the primitive type is the sample perfectile -- len = 96
                    extend_embedding_1.append(torch.from_numpy(embedding.reshape(-1)))
                    print("Sample Perfect Tile shape: ", extend_embedding_1[-1].shape)
                else: # prim type is other type: annotation & cuda bind -- len = 10
                    extend_embedding_0.append(torch.from_numpy(embedding.squeeze()))
                    print("Annotation & CUDA Bind shape: ", extend_embedding_0[-1].shape)
            # NOTE: Padding to max length for embeddings
            # TODO: Need padding condition
            MAX_NUMBER = 15
            # stack for convert [(10, ), (10, )..] into (15, 10) 
            if len(extend_embedding_0)>0:            
                extend_embedding_0 = torch.stack(extend_embedding_0, 0)
            else: # first shape is 15*10 -- binary vector
                extend_embedding_0 = torch.zeros(MAX_NUMBER,10)
            # stack for convert [(96, ), (96, )..] into (15, 96) 
            if len(extend_embedding_1)>0:
                extend_embedding_1 = torch.stack(extend_embedding_1, 0)
            else: # second shape is 15*96 -- binary vector
                extend_embedding_1 = torch.zeros(MAX_NUMBER,96)
            # NOTE: padding zeros, shape[1] is same 
            extend_embedding_0 = torch.cat([extend_embedding_0,
                                            torch.zeros(MAX_NUMBER - extend_embedding_0.shape[0],extend_embedding_0.shape[1]).to(extend_embedding_0.device)],0)
            extend_embedding_1 = torch.cat([extend_embedding_1,
                                            torch.zeros(MAX_NUMBER - extend_embedding_1.shape[0],extend_embedding_1.shape[1]).to(extend_embedding_1.device)],0)
            
            # Now extend_embedding_0's shape is (15,10), and extend_embedding_1's shape is (15,96)
            # After that, we flatten and concatenate them.            
            extend_embedding_0 = torch.argmax(extend_embedding_0,-1) # Translate one-hot to label (15, )
            extend_embedding_1 = extend_embedding_1.flatten() # Flatten it into (1440, )
            # Concatenate them, the last_embedding's shape is (15+15*96)
            last_embedding = torch.cat([extend_embedding_0,extend_embedding_1], 0) 
            # for embedding_condition in embedding_conditions:
            #     print("embedding condition shape: ", embedding_condition.shape)
            
            # NOTE: We do not need to translate it at a fine-grained level.
            last_condition = torch.cat([torch.from_numpy(embedding_condition.astype(int).reshape(-1)) for embedding_condition in embedding_conditions],0) 
            last_ptr_list = torch.Tensor(count_ptr_list)
            print("last embedding shape: ", last_embedding.shape) # torch.Size([1455])
            print("not padding condition shape: ", last_condition.shape) # torch.Size([72])
            print("last ptr list shape: ", last_ptr_list.shape) # torch.Size([3])
            # NOTE: We define attr in dataset: last_embedding, last_condition, last_ptr_lis, run_secs
            np.savez(os.path.join(save_path,f'mlc_{count_ptr}.npz'),last_embedding = last_embedding, last_condition = last_condition, last_ptr_list = last_ptr_list, run_secs = min_cost)
            print(f"Successfully Save File mlc_{count_ptr}.npz")
            count_ptr+=1
    
# dataset for GFlowNet: [15+15*96] 15 is 0~9 for anno+cuda
# GFlowNet prefer 0~9 format avoid invalid format [1, 0, 1] + extra mask -- GFlowNet output [0.3, 0.5, ..., 0.1] with 10 position
# 15*96 is 0~1 for tile (质因数分解) allowing [0, 1, 1] not one-hot format -- GFlowNet output [[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]]
class MLCGflowNetDataset(Dataset):
    # TODO: change without_condition=False & determine condition len in future 
    def __init__(self,all_files,without_condition=False, condition_embedding=200):
        self.all_files = all_files
        self.without_condition = without_condition
        self.condition_embedding = condition_embedding
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self,idx):
        file = self.all_files[idx]
        file = np.load(file)
        last_embedding = file['last_embedding']
        last_condition = file['last_condition']
        last_ptr_list = file['last_ptr_list']
        run_secs = file['run_secs']
            
        if self.without_condition:
            return last_embedding,run_secs
        else:
            padding = torch.zeros(self.condition_embedding - last_condition.shape[0]).to(last_condition.device)
            last_condition = torch.cat([last_condition,padding],0)
            return last_embedding,run_secs,last_condition,last_ptr_list
    
# Load above GFlowNet Dataset for search, ret dataloader
def gflownet_data_load(save_path,
                       without_condition=False,
                       num_workers=4,
                       batch_size=16,
                       shuffle=True,
                       pin_memory=False):
    import glob
    def search_all_files(work_dir):
        npz_files = sorted(glob.glob(os.path.join(work_dir, "*.npz"),recursive=True))
        return npz_files
    
    all_files = search_all_files(save_path)
    dataset = MLCGflowNetDataset(all_files,without_condition=without_condition)
    # A generative task without any validate dataset.
    # DataLoader speed up data loader
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,pin_memory=pin_memory)
    return dataloader