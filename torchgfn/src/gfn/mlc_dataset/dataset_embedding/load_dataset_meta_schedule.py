# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring

import argparse
import glob
import json
import os
from typing import List

import tvm
from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.ir import load_json
from tvm.target import Target
import multiprocessing
from multiprocessing.pool import ThreadPool
import copy


def search_all_files(work_dir):
    workload_files = glob.glob(os.path.join(
        work_dir, "**/*_workload.json"), recursive=True)
    results = []
    for workload_file in workload_files:
        candidate_file = workload_file.replace(
            "_workload.json", "_candidates.json")
        wc_pair = (workload_file, candidate_file)
        results.append(wc_pair)
    return results


def load_workload_and_candidate(result):
    workload_path, candidate_path = result
    database = ms.database.JSONDatabase(
        path_workload=workload_path, path_tuning_record=candidate_path)

    records = database.get_all_tuning_records()
    print(f"load len of records = {len(records)}")
    return database


def load_all_files0(work_dir):
    results = search_all_files(work_dir)
    databases = []
    # pool = multiprocessing.Pool(112)

    for result in results:
        database = load_workload_and_candidate(result)
        # database = load_workload_and_candidate(*result)
        databases.append(database)

    for database in databases:
        # database made up of records, including candidates info
        records = database.get_all_tuning_records()
        print(f"len of records = {len(records)}")
    return databases


def load_all_files(work_dir):

    results = search_all_files(work_dir)

    # NOTE: not work! len == 0
    # pool = multiprocessing.Pool(112)

    # NOTE: work! -- len > 0
    pool = ThreadPool(112)
    databases = pool.map(load_workload_and_candidate, results)
    pool.close()

    # NOTE: work! -- len > 0
    # databases = []
    # for result in results:
    #     database = load_workload_and_candidate(result)
    #     # database = load_workload_and_candidate(*result)
    #     databases.append(database)

    # for database in databases:
    #     # database made up of records, including candidates info
    #     records = database.get_all_tuning_records()
    #     print(f"len of records = {len(records)}")
    return databases


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir", type=str, help="Please provide the full path to the workload and candidate."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia/geforce-rtx-3070",
        help="Please specify the target hardware for tuning.\
                    Note: for generating dataset, the hardware does not need to be present.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()
    load_all_files(args.work_dir)
