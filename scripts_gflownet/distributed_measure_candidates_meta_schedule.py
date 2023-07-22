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
import os,copy

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.meta_schedule.builder import LocalBuilder, BuilderInput, BuilderResult
from tvm.meta_schedule.runner import LocalRunner, LocalRunnerFuture, RunnerInput, RunnerResult, RunnerFuture

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_cache_dir", type=str, help="Please provide the full path to the candidates."
    )
    parser.add_argument(
        "--result_cache_dir", type=str, help="Please provide the full path to the result database."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia/nvidia-a100",
        help="Please specify the target hardware for tuning context.",
    )
    parser.add_argument(
        "--builder_timeout_sec",
        type=int,
        default=10,
        help="The time for the builder session to time out.",
    )
    parser.add_argument(
        "--min_repeat_ms", type=int, default=100, help="The time for preheating the gpu."
    )
    parser.add_argument(
        "--runner_timeout_sec",
        type=int,
        default=100,
        help="The time for the runner session to time out.",
    )
    parser.add_argument(
        "--cpu_flush", type=bool, default=False, help="Whether to enable cpu cache flush or not."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size of candidates sent to builder and runner each time.",
    )
    return parser.parse_args()


# pylint: disable=too-many-locals
def measure_candidates(database, builder, runner):
    """Send the candidates to builder and runner for distributed measurement,
    and save the results in a new json database.

    Parameters
    ----------
    database : JSONDatabase
        The database for candidates to be measured.
    builder : Builder
        The builder for building the candidates.
    runner : Runner
        The runner for measuring the candidates.

    Returns
    -------
    None
    """
    candidates, runner_results, build_fail_indices, run_fail_indices = [], [], [], []
    context = ms.TuneContext(target=Target(args.target))
    tuning_records = database.get_all_tuning_records()
    for record in tuning_records:
        candidates.append(record.as_measure_candidate())
    with ms.Profiler() as profiler:
        for idx in range(0, len(candidates), args.batch_size):
            batch_candidates = candidates[idx : idx + args.batch_size]
            context.measure_candidates = batch_candidates  # pylint: disable=protected-access
            builder_inputs = []
            with ms.Profiler.timeit("build"):
                for candi in context.measure_candidates:
                    builder_inputs.append(BuilderInput(candi.sch.mod, context.target))
                context.builder_results= builder.build(builder_inputs)
            
            runner_inputs = []
            with ms.Profiler.timeit("run"):
                n_build_errors = 0
                candidates = context.measure_candidates
                builder_results = context.builder_results
                target = context.target
                for idx in range(len(builder_results)):
                    if builder_results[idx].error_msg is not None: #if build failed, continue
                        n_build_errors+=1
                        continue
                    runner_inputs.append(RunnerInput(builder_results[idx].artifact_path,
                                                target.kind.name,
                                                args_info=candidates[idx].args_info ))
                runner_futures = runner.run(runner_inputs)
                results = []
                if n_build_errors == 0:
                    context.runner_futures = runner_futures
                else:
                    j = 0
                    for builder_result in builder_results:
                        if builder_result.error_msg is not None:
                            results.append(RunnerFuture(lambda : True,lambda :RunnerResult(None, builder_result.error_msg)))
                        else:
                            results.append(runner_futures[j])
                            j+=1
                    context.runner_futures = results
                futures = context.runner_futures
                results =[future.result() for future in context.runner_futures]
                if context.search_strategy is not None:
                    context.search_strategy.notify_runner_results(context.measure_candidates,results)
                batch_runner_results = results
                for batch_runner_result in batch_runner_results:
                    print(batch_runner_result.error_msg,batch_runner_result.run_secs)
            runner_results.extend(batch_runner_results)
            for i, result in enumerate(context.builder_results):
                if result.error_msg is None:
                    ms.utils.remove_build_dir(result.artifact_path)
                else:
                    build_fail_indices.append(i + idx)
            context.measure_candidates = None
            context.bulder_results = None
            context.runner_futures = None

    model_name, workload_name = database.path_workload.split("/")[-2:]
    record_name = database.path_tuning_record.split("/")[-1]
    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(args.result_cache_dir, model_name, workload_name),
        path_tuning_record=os.path.join(args.result_cache_dir, model_name, record_name),
    )
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
    for i, (record, result) in enumerate(zip(tuning_records, runner_results)):
        if result.error_msg is None and result.run_secs is not None:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    run_secs=[v.value for v in result.run_secs],
                    target=Target(args.target),
                )
            )
        else:
            run_fail_indices.append(i)
    fail_indices_name = workload_name.replace("_workload.json", "_failed_indices.txt")
    with open(
        os.path.join(args.result_cache_dir, model_name, fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in run_fail_indices]))
    print(
        f"Builder time: {profiler.get()['build']}, Runner time: {profiler.get()['run']}\n\
            Failed number of builds: {len(build_fail_indices)},\
            Failed number of runs: {len(run_fail_indices)}"
    )


args = _parse_args()  # pylint: disable=invalid-name


def main():
    builder = ms.builder.LocalBuilder(timeout_sec=args.builder_timeout_sec)
    runner = ms.runner.LocalRunner(
        timeout_sec=100,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=args.min_repeat_ms,
            enable_cpu_cache_flush=args.cpu_flush,
        ),
        #max_workers=os.cpu_count(),
    )
    if not os.path.isdir(args.candidate_cache_dir):
        raise Exception("Please provide a correct candidate cache dir.")
    try:
        os.makedirs(args.result_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {args.result_cache_dir} cannot be created successfully.")
    model_dirs = glob.glob(os.path.join(args.candidate_cache_dir, "*"))
    for model_dir in model_dirs:
        model_name = model_dir.split("/")[-1]
        os.makedirs(os.path.join(args.result_cache_dir, model_name), exist_ok=True)
        all_tasks = glob.glob(os.path.join(model_dir, "*.json"))
        workload_paths = []
        for path in all_tasks:
            if path.endswith("_workload.json"):
                workload_paths.append(path)
        for workload_path in tqdm(workload_paths):
            candidate_path = workload_path.replace("_workload.json", "_candidates.json")
            database = ms.database.JSONDatabase(
                path_workload=workload_path,
                path_tuning_record=candidate_path,
            )
            measure_candidates(database, builder, runner)


if __name__ == "__main__":
    main()
