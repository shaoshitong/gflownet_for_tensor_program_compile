#! /bin/bash

mkdir ./dataset
python dataset_collect_models.py --model_cache_dir ./dataset/network_info
python dataset_extract_tasks.py --target "llvm -num-cores 56" --model_cache_dir ./dataset/network_info --task_cache_dir ./dataset/extract_tasks
python dataset_sample_candidates.py --target "llvm -num-cores 56" --task_cache_dir ./dataset/extract_tasks --candidate_cache_dir ./dataset/sample_candidate
python dataset_measure_candidates.py --target "llvm -num-cores 56" --candidate_cache_dir ./dataset/sample_candidate --result_cache_dir ./dataset/measure_candidate