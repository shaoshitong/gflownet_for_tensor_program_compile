from .load_dataset_meta_schedule import load_all_files
import argparse
import glob
import json
import os
from typing import List, Tuple, Dict
import copy

import tvm
from dataclasses import dataclass
from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.ir import load_json
from tvm.target import Target
from tvm.tir.schedule import Trace
from tvm.tir.schedule import InstructionKind, Instruction
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
        InstructionKind.get("SampleCategorical"): "SampleCategorical",
        InstructionKind.get("Split"): "Split",
        InstructionKind.get("Bind"): "Bind",
        InstructionKind.get("SamplePerfectTile"): "SamplePerfectTile",
        InstructionKind.get("Annotate"): "Annotate",
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
class EmbeddingAnnotation:

    embedding_len = 10

    # Now, we define the "meta_schedule.cooperative_fetch" as 0, \
    # "meta_schedule.unroll_explicit" as 1, "meta_schedule.unroll_implicit" as

    @staticmethod
    def embedding_annotation(insts, decisions) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        new_insts = []
        new_decis = []
        embedding_results = []
        embedding_conditions = []
        annotations = set()
        sample_insts = {}
        ann_insts = []
        # NOTE: only care `Annotate` `SampleCategorical`
        for sub_inst in insts:
            if sub_inst.kind == InstructionKind.get("Annotate"):
                name = sub_inst.attrs[0]
                if str(name) == "meta_schedule.cooperative_fetch":
                    ann_val = sub_inst.inputs[1]
                    annotations.add(ann_val)
                # ann_insts: [annotate(unroll..),...]
                elif str(name) == "meta_schedule.unroll_explicit" or str(name) == "meta_schedule.unroll_implicit":
                    ann_insts.append(sub_inst)
            # sample_insts['v8'] = sch.SampleCategorical - v8 is `var_rv`
            elif sub_inst.kind == InstructionKind.get("SampleCategorical"):
                sample_insts[sub_inst.outputs[0]] = sub_inst

        for sub_inst, sub_value in decisions.items():
            if sub_inst.kind == InstructionKind.get("SampleCategorical"):
                if sub_inst.outputs[0] in annotations:
                    probs = [i.value for i in sub_inst.attrs[1]]
                    values = [i.value for i in sub_inst.attrs[0]]
                    if len(probs) == 1:
                        continue
                    new_value = sub_value.value
                    one_hot = np.zeros(
                        (EmbeddingAnnotation.embedding_len,), dtype=np.float32)
                    one_hot[new_value] = 1
                    # old_len is valid len of probs & vals
                    old_len = len(probs)
                    while len(probs) < EmbeddingAnnotation.embedding_len:
                        probs += [0]
                    while len(values) < EmbeddingAnnotation.embedding_len:
                        values += [0]
                    condition = [0, 0, 0] + [old_len] + \
                        probs + values  # fix len: first 3 is id
                    new_insts.append(str(sub_inst))
                    new_decis.append(int(sub_value))
                    embedding_conditions.append(np.array(condition))
                    embedding_results.append(one_hot)

        for ann_inst in ann_insts:
            # ["Annotate",["l87",1],["pragma_unroll_explicit"],[]]
            var_rv = ann_inst.inputs[1]
            sample_categorical = sample_insts[var_rv]
            new_value = decisions[sample_categorical].value
            probs = [i.value for i in sample_categorical.attrs[1]]
            values = [i.value for i in sample_categorical.attrs[0]]

            one_hot = np.zeros(
                (EmbeddingAnnotation.embedding_len,), dtype=np.float32)
            one_hot[new_value] = 1
            old_len = len(probs)
            while len(probs) < EmbeddingAnnotation.embedding_len:
                probs += [0]
            while len(values) < EmbeddingAnnotation.embedding_len:
                values += [0]
            if str(ann_inst.attrs[0]) == "meta_schedule.unroll_explicit":
                condition = [1, 0, 0] + [old_len] + probs + values
            elif str(ann_inst.attrs[0]) == "meta_schedule.unroll_implicit":
                condition = [0, 1, 0] + [old_len] + probs + values
            else:
                raise NotImplementedError
            new_insts.append(str(sample_categorical))
            new_decis.append(int(decisions[sample_categorical]))
            embedding_conditions.append(np.array(condition))
            embedding_results.append(one_hot)
        # res: one-hot, condition:
        return embedding_results, embedding_conditions, new_insts, new_decis

    @staticmethod
    def unembedding_annotation(insts, decisions, embedding_results, embedding_conditions) -> Tuple[List[Instruction], Dict[Instruction, int]]:

        # new_insts = []
        # Same order for insts and decisions
        decisions = dict(decisions)
        new_decisions = decisions
        count_ptr = 0
        if len(embedding_results) == 0:
            return new_decisions

        annotations = set()
        sample_insts = {}
        ann_insts = []
        for sub_inst in insts:
            if sub_inst.kind == InstructionKind.get("Annotate"):
                name = sub_inst.attrs[0]
                if str(name) == "meta_schedule.cooperative_fetch":
                    ann_val = sub_inst.inputs[1]
                    annotations.add(ann_val)

                elif str(name) == "meta_schedule.unroll_explicit" or str(name) == "meta_schedule.unroll_implicit":
                    ann_insts.append(sub_inst)

            elif sub_inst.kind == InstructionKind.get("SampleCategorical"):
                sample_insts[sub_inst.outputs[0]] = sub_inst

        for sub_inst, sub_value in decisions.items():
            if sub_inst.kind == InstructionKind.get("SampleCategorical"):
                if sub_inst.outputs[0] in annotations:
                    embedding_result = embedding_results[count_ptr]
                    # 判断变量是否为ndarray
                    if len(embedding_result.shape) > 0:
                        new_value = int(np.argmax(embedding_result))
                    else:
                        new_value = int(embedding_result)
                    new_value = tvm.tir.const(new_value, dtype='int32')
                    # new_insts.append(sub_inst)
                    new_decisions[sub_inst] = new_value
                    count_ptr += 1

        for ann_inst in ann_insts:
            # ["Annotate",["l87",1],["pragma_unroll_explicit"],[]]
            var_rv = ann_inst.inputs[1]
            sample_categorical = sample_insts[var_rv]
            embedding_result = embedding_results[count_ptr]
            if len(embedding_result.shape) > 0:
                new_value = int(np.argmax(embedding_result))
            else:
                new_value = int(embedding_result)

            new_value = tvm.tir.const(new_value, dtype='int32')
            # new_insts.append(sample_categorical)
            new_decisions[sample_categorical] = new_value
            count_ptr += 1

        # if len(new_insts) == len(list(decisions.values())):
        #     print(f"Same len for old decision & new decision in annotation")
        return new_decisions