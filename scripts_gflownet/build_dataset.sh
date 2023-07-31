#!/bin/bash

mkdir ./dataset
python dump_network_info_meta_schedule.py --model_cache_dir ./dataset/network_info
python dataset_extract_tasks_meta_schedule.py --target "cuda --max_threads_per_block 1024 --thread_warp_size 32 --max_shared_memory_per_block 49152" --model_cache_dir ./dataset/network_info --task_cache_dir ./dataset/extract_tasks
python dataset_sample_candidates_meta_schedule.py --target "nvidia/nvidia-a100" --task_cache_dir ./dataset/extract_tasks --candidate_cache_dir ./dataset/sample_candidate
CUDA_VISIBLE_DEVICES=1 python distributed_measure_candidates_meta_schedule.py --target "nvidia/nvidia-a100" --candidate_cache_dir ./dataset/sample_candidate --result_cache_dir ./dataset/measure_candidate_2