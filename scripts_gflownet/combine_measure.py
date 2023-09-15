from dataset_embedding import load_all_files,check_decision_same,GflowNetEmbedding
import tvm
from tvm.meta_schedule.module_equality import ModuleEquality
databases_2 = load_all_files("/home/tvm/scripts_gflownet/dataset/measure_candidate_2")

hash_dict = {}

for database in databases_2:
    records = database.get_all_tuning_records()
    for record in records:
        sub_sch = record.as_measure_candidate().sch
        sub_mod = sub_sch.mod
        sub_trace = sub_sch.trace
        sub_insts = sub_trace.insts
        sub_decisions = sub_trace.decisions
        run_secs = record.run_secs
        hash_mod = ModuleEquality("structural").hash(sub_mod)
        hash_dict[hash_mod] = run_secs
        
    
databases = load_all_files("/home/tvm/scripts_gflownet/dataset/measure_candidate")

import os

import tvm.meta_schedule as ms
for database in databases:
    
    os.makedirs(os.path.join("/home/tvm/scripts_gflownet/dataset/measure_candidate_3/", database.path_workload.split("/")[-2]), exist_ok=True)
    new_database = ms.database.JSONDatabase(
        path_workload=database.path_workload.replace("measure_candidate","measure_candidate_3"),
        path_tuning_record=database.path_tuning_record.replace("measure_candidate","measure_candidate_3"),
    )
    
    records = database.get_all_tuning_records()
    for i,record in enumerate(records):
        sub_sch = record.as_measure_candidate().sch
        sub_mod = sub_sch.mod
        sub_trace = sub_sch.trace
        sub_insts = sub_trace.insts
        sub_decisions = sub_trace.decisions
        run_secs = record.run_secs
        hash_mod = ModuleEquality("structural").hash(sub_mod)
        
        workload = record.workload
        new_database.commit_workload(workload.mod)
        
        if hash_mod in hash_dict:
            new_run_secs = []
            for num in run_secs:
                new_run_secs.append(num)
            for num in hash_dict[hash_mod]:
                new_run_secs.append(num)
            run_secs = new_run_secs
        else:
            new_run_secs = []
            for num in run_secs:
                new_run_secs.append(num)
            run_secs = new_run_secs
                     
        new_database.commit_tuning_record(
            ms.database.TuningRecord(
                trace=record.trace,
                workload=workload,
                run_secs=run_secs,
                target=record.target,
            )
        )
        