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

def get_useful_keys():
    KEYS = [
        InstructionKind.get("SampleCategorical"),
        InstructionKind.get("Split"),
        InstructionKind.get("Bind"),
        InstructionKind.get("SamplePerfectTile"),
        InstructionKind.get("Annotate"),   
    ]
    return KEYS

def get_useful_dicts():
    KEYS_VALUES = {
        InstructionKind.get("SampleCategorical") : "SampleCategorical",
        InstructionKind.get("Split"): "Split",
        InstructionKind.get("Bind") : "Bind",
        InstructionKind.get("SamplePerfectTile") : "SamplePerfectTile",
        InstructionKind.get("Annotate") : "Annotate",   
    }
    return KEYS_VALUES

def deep_copy_map(old_map):
    # Create a new map with a generator that copies the items from the old map
    new_map = ({k: v for k, v in old_map.items()})
    return new_map

def deep_copy_array(old_array):
    new_array = ([item for item in old_array])
    return new_array

@dataclass
class EmbeddingCUDABind:

    embedding_len = 10
    
    @staticmethod
    def is_split_by_sample(sub_inst,sample_insts):
        if sub_inst.kind != InstructionKind.get("Split"):
            return False
        if len(sub_inst.inputs)!=3 or sub_inst.inputs[1] is not None:
            return False
        if sub_inst.inputs[2] in sample_insts:
            return True
        return False
    
    @staticmethod
    def is_thread_binding_by_sample(sub_inst,sampled_split_insts):
        if sub_inst.kind != InstructionKind.get("Bind"):
            return False
        if sub_inst.attrs[0] !="threadIdx.x":
            return False
        if sub_inst.inputs[0] in sampled_split_insts:
            return True
        return False
        
    @staticmethod
    def embedding_cudabind(insts,decisions) -> Tuple[List[np.ndarray],List[np.ndarray]]:
        embedding_results = []
        embedding_conditions = []
        
        sample_insts = {}
        sampled_split_insts = {}
        bind_insts = []
        for sub_inst in insts:
            if sub_inst.kind == InstructionKind.get("SampleCategorical"): # sub_inst is SampleCategorical
                var_rv = sub_inst.outputs[0]
                sample_insts[var_rv] = sub_inst
            elif EmbeddingCUDABind.is_split_by_sample(sub_inst,sample_insts): # sub_inst is Split
                var_rv = sub_inst.outputs[1]
                sampled_split_insts[var_rv] = sub_inst
            elif EmbeddingCUDABind.is_thread_binding_by_sample(sub_inst,sampled_split_insts):
                bind_insts.append(sub_inst)
                
        for bind_inst in bind_insts:
            loop_rv = bind_inst.inputs[0]
            split_inst = sampled_split_insts[loop_rv]
            expr_rv = split_inst.inputs[2]
            sample_inst = sample_insts[expr_rv]
            decision = decisions[sample_inst].value
            probs = [i.value for i in sample_inst.attrs[1]]
            values = [i.value for i in sample_inst.attrs[0]]
            
            one_hot = np.ones((EmbeddingCUDABind.embedding_len,),dtype=np.float32)
            one_hot[decision] = 1
            old_len = len(probs)

            while len(probs)<EmbeddingCUDABind.embedding_len:
                probs+=[0]
            while len(values)<EmbeddingCUDABind.embedding_len:
                values+=[0]
            condition = [0,0,1] + [old_len] + probs + values
            
            
            embedding_conditions.append(np.array(condition))
            embedding_results.append(one_hot)
        
        return embedding_results,embedding_conditions
        
    @staticmethod
    def unembedding_cudabind(insts,decisions,embedding_results,embedding_conditions) -> Tuple[List[Instruction],Dict[Instruction,int]]:
        
        sample_insts = {}
        sampled_split_insts = {}
        bind_insts = []
        for sub_inst in insts:
            if sub_inst.kind == InstructionKind.get("SampleCategorical"): # sub_inst is SampleCategorical
                var_rv = sub_inst.outputs[0]
                sample_insts[var_rv] = sub_inst
            elif EmbeddingCUDABind.is_split_by_sample(sub_inst,sample_insts): # sub_inst is Split
                var_rv = sub_inst.outputs[1]
                sampled_split_insts[var_rv] = sub_inst
            elif EmbeddingCUDABind.is_thread_binding_by_sample(sub_inst,sampled_split_insts):
                bind_insts.append(sub_inst)
    
        count_ptr = 0
        new_insts = []
        new_decisions = []
        
        for bind_inst in bind_insts:
            loop_rv = bind_inst.inputs[0]
            split_inst = sampled_split_insts[loop_rv]
            expr_rv = split_inst.inputs[2]
            sample_inst = sample_insts[expr_rv]   
            one_hot = embedding_results[count_ptr]
            count_ptr+=1
            new_value = np.argmax(one_hot)
            new_value = tvm.tir.const(new_value, dtype='int32')
            new_insts.append(sample_inst)
            new_decisions.append(new_value)
        
        return new_insts,new_decisions