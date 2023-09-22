import tvm,os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tvm.meta_schedule.cost_model.tlp_cost_model_train import from_json,load_data
from .dataset_embedding import GflowNetEmbedding, load_all_files

# To make a GFlowNet dataset
def gflownet_data_save(data_path,save_path):
    assert os.path.exists(data_path), f"{data_path} not exists!"
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
        records = database.get_all_tuning_records()
        for record in records:
            if os.path.exists(f"mlc_{count_ptr}.npz"):
                count_ptr+=1
                print(f"Passing mlc_{count_ptr}.npz")
                continue
            sub_sch = record.as_measure_candidate().sch
            min_cost = _min_cost(record)
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions
            
            extend_embedding_0 = []
            extend_embedding_1 = []
            embedding_results,embedding_conditions,count_ptr_list = gm(sub_insts,sub_decisions,True)
            for embedding in embedding_results:
                _len = embedding.shape[0]
                if _len > 10: # If the primitive type is the sample perfectile
                    extend_embedding_1.append(torch.from_numpy(embedding.reshape(-1)))
                    print(extend_embedding_1[-1].shape,"9")
                else:
                    extend_embedding_0.append(torch.from_numpy(embedding.squeeze()))
                    print(extend_embedding_0[-1].shape,"11")
            # NOTE: Padding to max length for embeddings
            # TODO: Need padding condition
            MAX_NUMBER = 15
            if len(extend_embedding_0)>0:            
                extend_embedding_0 = torch.stack(extend_embedding_0, 0)
            else:
                extend_embedding_0 = torch.zeros(MAX_NUMBER,10)
            if len(extend_embedding_1)>0:
                extend_embedding_1 = torch.stack(extend_embedding_1, 0)
            else:
                extend_embedding_1 = torch.zeros(MAX_NUMBER,96)

            extend_embedding_0 = torch.cat([extend_embedding_0,
                                            torch.zeros(MAX_NUMBER - extend_embedding_0.shape[0],extend_embedding_0.shape[1]).to(extend_embedding_0.device)],0)
            extend_embedding_1 = torch.cat([extend_embedding_1,
                                            torch.zeros(MAX_NUMBER - extend_embedding_1.shape[0],extend_embedding_1.shape[1]).to(extend_embedding_1.device)],0)
            
            # Now extend_embedding_0's shape is (15,10), and extend_embedding_1's shape is (15,96)
            # After that, we flatten and concatenate them.
            
            extend_embedding_0 = torch.argmax(extend_embedding_0,-1) # Translate one-hot to label
            extend_embedding_1 = extend_embedding_1.flatten() # Flatten it
            last_embedding = torch.cat([extend_embedding_0,extend_embedding_1],0) # Concatenate them, the last_embedding's shape is (15+15*96)
            for embedding_condition in embedding_conditions:
                print(embedding_condition.shape)
            
            last_condition = torch.cat([torch.from_numpy(embedding_condition.astype(int).reshape(-1)) for embedding_condition in embedding_conditions],0) # We do not need to translate it at a fine-grained level.
            last_ptr_list = torch.Tensor(count_ptr_list)
            print(last_embedding.shape,last_condition.shape,last_ptr_list.shape)
            # NOTE: We define attr in dataset 
            np.savez(os.path.join(save_path,f'mlc_{count_ptr}.npz'),last_embedding = last_embedding, last_condition = last_condition, last_ptr_list = last_ptr_list, run_secs = min_cost)
            print(f"Successfully Save File mlc_{count_ptr}.npz")
            count_ptr+=1
    
# dataset for GFlowNet: [15+15*96] 15 is 0~9 for anno+cuda
# GFlowNet prefer 0~9 format avoid invalid format [1, 0, 1] + extra mask -- GFlowNet output [0.3, 0.5, ..., 0.1] with 10 position
# 15*96 is 0~1 for tile (质因数分解) allowing [0, 1, 1] not one-hot format -- GFlowNet output [[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]]
class MLCGflowNetDataset(Dataset):
    # TODO: change without_condition=False & determine condition len in future 
    def __init__(self,all_files,without_condition=True, condition_embedding=200):
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
            padding = torch.zeros(self.last_condition - last_condition.shape[0]).to(last_condition.device)
            last_condition = torch.cat([last_condition,padding],0)
            return last_embedding,run_secs,last_condition,last_ptr_list
    
# Load above GFlowNet Dataset
def gflownet_data_load(save_path,
                       without_condition=True,
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