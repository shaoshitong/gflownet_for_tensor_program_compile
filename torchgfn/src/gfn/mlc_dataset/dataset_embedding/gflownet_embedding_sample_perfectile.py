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
class EmbeddingSamplePerfectTile:

    embedding_len = 32

    embedding_total = 7

    @staticmethod
    def pollards_rho(n):
        if n % 2 == 0:
            return 2
        x = np.random.randint(2, n)
        y = x
        c = np.random.randint(1, n)
        g = 1
        while g == 1:
            x = ((x * x) % n + c + n) % n
            y = ((y * y) % n + c + n) % n
            y = ((y * y) % n + c + n) % n
            g = np.gcd(abs(x-y), n)
        return g

    @staticmethod
    def prime_factors(n):
        factors = []
        while n % 2 == 0:
            factors.append(2),
            n = n // 2
        while n != 1:
            factor = EmbeddingSamplePerfectTile.pollards_rho(n)
            factors.append(factor)
            n = n // factor
        return np.array(factors)

    @staticmethod
    def expand_to_binary(array):
        max_len = int(
            np.ceil(np.log2(EmbeddingSamplePerfectTile.embedding_total + 1)))
        binary_array = np.zeros(array.shape + (max_len,), dtype=np.uint8)
        for idx in np.ndindex(array.shape):
            binary_array[idx] = list(f'{array[idx]:0{max_len}b}')[::-1]
        return binary_array

    @staticmethod
    def convert_to_original(binary_array):
        # max_len = int(np.ceil(np.log2(EmbeddingSamplePerfectTile.embedding_total + 1)))
        original_shape = binary_array.shape[-1]
        original_array = np.zeros(original_shape, dtype=int)

        for idx in np.ndindex(original_shape):
            binary_str = ''.join(map(str, binary_array[idx][::-1]))
            original_array[idx] = int(binary_str, 2)

        return original_array

    @staticmethod
    def embedding_sample_perfectile(insts, decisions) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        This is a lossless encoding function for tiles, he expects to get multiple tile sample \
        classes from insts as well as decisions and then and encode them. Note that only the \
        rules need to be uniquely determined, the DECISION of the tile sample does not need to \
        be guaranteed.

        Returns:
            tuple[list[np.ndarray],list[np.ndarray]]: An Embedding only with 0/1
        """

        embedding_results = []
        embedding_conditions = []
        for sub_inst, sub_value in decisions.items():
            if sub_inst.kind == InstructionKind.get("SamplePerfectTile"):
                np_sub_value = np.array([v.value for v in sub_value])
                # origin num of tile 7: (4, 1, 32, 1, 1, 1, 1)
                nums = sub_inst.attrs[0]
                max_innermost_factor = sub_inst.attrs[1]  # 64
                # How to embedding it?
                # First, compute the prod of all factors
                np_prod_value = np.prod(np_sub_value)

                # Second, decompose prime factors (math.)
                factors = EmbeddingSamplePerfectTile.prime_factors(
                    np_prod_value)
                masks = [False for _ in range(len(factors))]
                embedding_result = [0 for _ in range(
                    EmbeddingSamplePerfectTile.embedding_len)]  # len(32)

                # Third, embedding to index - map index of all factors to pos
                # origin tiles (4, 1, 32, 1, 1), factors (2, 2, 2, 2, 2, 2, 2)
                # res (1, 1, 3, 3, 3, 3, 3, 0, 0, ...)
                for i, np_sub_v in enumerate(np_sub_value.tolist()):
                    if np_sub_v == 1:
                        continue
                    c_np_sub_v = np_sub_v
                    for j, factor in enumerate(factors):
                        if c_np_sub_v % factor == 0 and masks[j] == False:
                            c_np_sub_v = c_np_sub_v // factor
                            embedding_result[j] = i+1
                            masks[j] = True
                # Fourth, embedding in binary - (32, 3) binary value
                embedding_result = EmbeddingSamplePerfectTile.expand_to_binary(
                    np.array(embedding_result))
                embedding_results.append(embedding_result)

                # Fifth, embedding the condition
                embedding_condition = []
                embedding_condition.append(nums)
                embedding_condition.append(max_innermost_factor)
                embedding_condition += factors.tolist()
                while len(embedding_condition) < EmbeddingSamplePerfectTile.embedding_len+2:
                    embedding_condition += [0]  # padding to 34
                embedding_condition = np.array(embedding_condition).astype(int)
                embedding_conditions.append(embedding_condition)
        return embedding_results, embedding_conditions

    @staticmethod
    def unembedding_sample_perfectile(insts, decisions, embedding_results, embedding_conditions) -> Tuple[List[Instruction], Dict[Instruction, int]]:

        new_insts = []
        new_decisions = []
        if len(embedding_results) == 0:
            return new_insts, new_decisions

        count_ptr = 0
        for sub_inst, sub_value in decisions.items():
            if sub_inst.kind == InstructionKind.get("SamplePerfectTile"):
                np_sub_value = np.array([v.value for v in sub_value])
                new_np_sub_value = np.ones_like(np_sub_value)

                embedding_result = embedding_results[count_ptr]
                embedding_result = EmbeddingSamplePerfectTile.convert_to_original(
                    embedding_result).tolist()  # shape (32,)

                embedding_condition = embedding_conditions[count_ptr]
                # first 2 is num & max_factor, last is padding zeros
                factors = embedding_condition[2:]
                first_zero_index = np.argmin(factors != 0)
                factors = factors[:first_zero_index]

                # Unembedding
                for i in range(len(embedding_result)):
                    if embedding_result[i] != 0:
                        new_np_sub_value[embedding_result[i]-1] *= factors[i]
                new_insts.append(sub_inst)
                new_decisions.append([tvm.tir.const(i, dtype='int32')
                                     for i in new_np_sub_value.tolist()])
                count_ptr += 1
        return new_insts, new_decisions
