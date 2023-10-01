from .gflownet_embedding_annotation import EmbeddingAnnotation
from .gflownet_embedding_cuda_bind import EmbeddingCUDABind
from .gflownet_embedding_sample_perfectile import EmbeddingSamplePerfectTile
from .load_dataset_meta_schedule import load_all_files
import argparse
import glob
import json
import os
from typing import List,Tuple,Dict
import copy

import tvm
from dataclasses import dataclass
from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.ir import load_json
from tvm.target import Target
from tvm.tir.schedule import Trace
from tvm.tir.schedule import InstructionKind,Instruction
import numpy as np
from tvm.tir.schedule import InstructionKind
from mlc_dataset.dataset_embedding import deep_copy_map

def check_decision_same(decisions1,decisions2):
    for decision1,decision2 in zip(decisions1,decisions2):
        if type(decision1)!=type(decision2):
            print("types are not same",type(decision1),type(decision2))
            return False
        if isinstance(decision1,(int,tvm.tir.expr.IntImm)):
            if decision1!=decision2:
                print("ints are not same",decision1,decision2)
                return False
        elif isinstance(decision1,(list,tvm.ir.container.Array)):
            check = check_decision_same(decision1,decision2)
            if check == False:
                print("lists are not same",decision1,decision2)
                return False
        else:
            print("unknown type",type(decision1),type(decision2))
            return False
    return True
            
@dataclass
class GflowNetEmbedding:
    
    def __init__(self,):
        pass
    
    check = check_decision_same
    e_annotation = EmbeddingAnnotation
    e_cuda_bind = EmbeddingCUDABind
    e_sample_perfectile = EmbeddingSamplePerfectTile
    
    @staticmethod
    def embedding(insts,decisions) -> Tuple[List[np.ndarray],List[np.ndarray],List[int]]:
        
        count_ptr = 0
        count_Ptr_results = []
        embedding_results,embedding_conditions = [],[]
        _embedding_results,_embedding_conditions = GflowNetEmbedding.e_annotation.embedding_annotation(insts,decisions)
        embedding_results += _embedding_results
        embedding_conditions += _embedding_conditions
        count_ptr += len(_embedding_results)
        count_Ptr_results.append(count_ptr)
        
        _embedding_results,_embedding_conditions = GflowNetEmbedding.e_cuda_bind.embedding_cudabind(insts,decisions)
        embedding_results += _embedding_results
        embedding_conditions += _embedding_conditions
        count_ptr += len(_embedding_results)
        count_Ptr_results.append(count_ptr)
        
        _embedding_results,_embedding_conditions = GflowNetEmbedding.e_sample_perfectile.embedding_sample_perfectile(insts,decisions)
        embedding_results += _embedding_results
        embedding_conditions += _embedding_conditions
        count_ptr += len(_embedding_results)
        count_Ptr_results.append(count_ptr)
        # condition is input for constraints of GFlowNet, also for decoding (include prim type)
        # result is binary vector: 15*10 (15 is annotation+cuda sample number, 10 is sample result index range [0, 9]) + 15*96
        # (15 is sample perfect tile number, 96 is 质因数分解 number)
        # TODO: input condition into GFlowNet
        return embedding_results,embedding_conditions,count_Ptr_results
    
    @staticmethod
    def unembedding(insts, decisions, embedding_results, embedding_conditions, count_Ptr_results) -> Tuple[List[Instruction],Dict[Instruction,int]]:
        
        previous_count_ptr = 0
        type_list = [GflowNetEmbedding.e_annotation.unembedding_annotation,GflowNetEmbedding.e_cuda_bind.unembedding_cudabind,GflowNetEmbedding.e_sample_perfectile.unembedding_sample_perfectile]
        new_insts = []
        new_decisions = []
        for v,i in enumerate(count_Ptr_results):
            _embedding_results = embedding_results[previous_count_ptr:i]
            _embedding_conditions = embedding_conditions[previous_count_ptr:i]
            previous_count_ptr = i
            _new_insts,_new_decisions = type_list[v](insts,decisions,_embedding_results,_embedding_conditions)
            new_insts += _new_insts
            new_decisions += _new_decisions
        
        return new_insts,new_decisions
    
    def __call__(self,insts,decisions,if_embedding,**kwargs):
        """
        Args:
            insts: instructions of the schedule
            decisions: decisions of the schedule
            if_embedding: apply embedding or unembedding
            Optional[embedding_results]: only use in unembedding
            Optional[embedding_conditions]: only use in unembedding
            Optional[count_Ptr_results]: only use in unembedding
        """
        
        if if_embedding:
            return GflowNetEmbedding.embedding(insts,decisions)
        else:
            assert "embedding_results" in kwargs, "embedding_results is not in kwargs"
            assert "embedding_conditions" in kwargs, "embedding_conditions is not in kwargs"
            assert "count_Ptr_results" in kwargs, "count_Ptr_results is not in kwargs"
            return GflowNetEmbedding.unembedding(insts,decisions,embedding_conditions=kwargs["embedding_conditions"],embedding_results=kwargs["embedding_results"],count_Ptr_results=kwargs["count_Ptr_results"])
        
    def check(self,insts,decisions):
        embedding_results,embedding_conditions,count_Ptr_results = self(insts,decisions,True)
        new_insts,new_decisions = self(insts,decisions,False,embedding_results=embedding_results,embedding_conditions=embedding_conditions,count_Ptr_results=count_Ptr_results)
        AA = list(deep_copy_map(decisions).values())
        BB = list(new_decisions.values())
        return check_decision_same(AA,BB)