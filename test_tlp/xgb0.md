2023-10-17 15:31:28 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 15:31:32 [INFO] LocalBuilder: max_workers = 56
2023-10-17 15:31:33 [INFO] LocalRunner: max_workers = 1
2023-10-17 15:36:09 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 15:36:09 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 15:36:12 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 15:36:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 15:41:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x531a108)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5378c88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x525eab8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5280908)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x531b868)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x525d3e8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x525e2d8)]: 0 failure(s)
2023-10-17 15:41:19 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 15:41:19 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 15:41:20 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 15:41:28 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 15:41:28 [INFO] LocalBuilder: max_workers = 56
2023-10-17 15:41:30 [INFO] LocalRunner: max_workers = 1
2023-10-17 15:46:06 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 15:46:06 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 15:46:06 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 15:46:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 15:51:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 26 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x1442bf3b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x143f554d8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144120858)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x143f6dac8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144058628)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1441cc5c8)]: 26 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x144057d58)]: 0 failure(s)
2023-10-17 15:51:18 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 15:51:18 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 15:51:18 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 15:51:24 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 15:51:24 [INFO] LocalBuilder: max_workers = 56
2023-10-17 15:51:25 [INFO] LocalRunner: max_workers = 1
2023-10-17 15:56:00 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 15:56:00 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 15:56:01 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 15:56:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:01:11 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:11 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:13 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:13 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa990118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xb02a1c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb38d7e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa9412f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8b70b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1499af008)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14981c858)]: 0 failure(s)
2023-10-17 16:01:13 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:01:13 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:01:13 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:01:19 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:01:19 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:01:21 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:05:56 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:05:56 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:05:57 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:05:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:11:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb270918)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14678d4b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1484044d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5033df8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb294ae8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1483d5638)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499838b8)]: 0 failure(s)
2023-10-17 16:11:05 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:11:05 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:11:05 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:11:11 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:11:11 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:11:13 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:15:47 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:15:47 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:15:47 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:15:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:21:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb3e45c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14410f1b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x8369158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xaecf948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae0ca98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14868a268)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486a5e78)]: 0 failure(s)
2023-10-17 16:21:03 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:21:03 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:21:03 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:21:11 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:21:11 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:21:13 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:26:22 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:26:22 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:26:22 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:26:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:31:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x148714e18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1464ce278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb6bd6d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1494b9878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c5ccc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d392608)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1464ca6f8)]: 0 failure(s)
2023-10-17 16:31:30 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:31:31 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:31:31 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:31:39 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:31:39 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:31:42 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:36:56 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:36:56 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:36:56 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:36:56 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:42:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14401c9d8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14820e588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7d1448)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148403768)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x9a1ad98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c0dfb8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14c2930c8)]: 0 failure(s)
2023-10-17 16:42:04 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:42:05 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:42:05 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:42:12 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:42:12 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:42:14 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:47:33 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:47:33 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:47:33 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:47:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 16:52:45 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:46 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:46 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xad0d228)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14839de28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146bdb0e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149906078)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa917c88)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x144982cd8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5486268)]: 0 failure(s)
2023-10-17 16:52:48 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 16:52:49 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 16:52:49 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 16:52:57 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 16:52:57 [INFO] LocalBuilder: max_workers = 56
2023-10-17 16:52:59 [INFO] LocalRunner: max_workers = 1
2023-10-17 16:58:05 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 16:58:05 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 16:58:06 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 16:58:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 17:04:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x146d63a48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x53b3ee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1497ae4d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d48abb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14646aef8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147725c88)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14d64d4b8)]: 0 failure(s)
2023-10-17 17:04:06 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 17:04:07 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 17:04:07 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

2023-10-17 17:04:18 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 17:04:18 [INFO] LocalBuilder: max_workers = 56
2023-10-17 17:04:20 [INFO] LocalRunner: max_workers = 1
2023-10-17 17:09:35 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 17:09:35 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 17:09:35 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 17:09:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 17:17:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c261bc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14c20b228)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x146a7ff38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1477b7968)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5485ee8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146017838)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147bcef38)]: 0 failure(s)
2023-10-17 17:17:31 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 17:17:32 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 17:17:32 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006 ms
[90;03m# from tvm import tir[39;00m
[32;01mdef[39;00m [34;01mapply_trace[39;00m(sch: tir[35;01m.[39;00mSchedule) [35;01m-[39;00m[35;01m>[39;00m [32;01mNone[39;00m:
    b0 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    b1 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.tiling_structure[39m[33m"[39m, ann_val[35;01m=[39;00m[33m"[39m[33mSSSRRSRS[39m[33m"[39m
    )
    l2, l3, l4 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb0)
    v5, v6, v7, v8, v9 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml2, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m16[39m, [92m2[39m, [92m4[39m, [92m1[39m, [92m1[39m]
    )
    l10, l11, l12, l13, l14 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml2, factors[35;01m=[39;00m[v5, v6, v7, v8, v9], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v15, v16, v17, v18, v19 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml3, n[35;01m=[39;00m[92m5[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m4[39m, [92m1[39m, [92m32[39m, [92m1[39m, [92m1[39m]
    )
    l20, l21, l22, l23, l24 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml3, factors[35;01m=[39;00m[v15, v16, v17, v18, v19], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    v25, v26, v27 [35;01m=[39;00m sch[35;01m.[39;00msample_perfect_tile(
        loop[35;01m=[39;00ml4, n[35;01m=[39;00m[92m3[39m, max_innermost_factor[35;01m=[39;00m[92m64[39m, decision[35;01m=[39;00m[[92m2[39m, [92m4[39m, [92m16[39m]
    )
    l28, l29, l30 [35;01m=[39;00m sch[35;01m.[39;00msplit(
        loop[35;01m=[39;00ml4, factors[35;01m=[39;00m[v25, v26, v27], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m
    )
    sch[35;01m.[39;00mreorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l10, l20, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml31, thread_axis[35;01m=[39;00m[33m"[39m[33mblockIdx.x[39m[33m"[39m)
    l32 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l11, l21, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml32, thread_axis[35;01m=[39;00m[33m"[39m[33mvthread.x[39m[33m"[39m)
    l33 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l12, l22, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml33, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_low_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m32[39m,
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb0,
        ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.thread_extent_high_inclusive[39m[33m"[39m,
        ann_val[35;01m=[39;00m[92m1024[39m,
    )
    b34 [35;01m=[39;00m sch[35;01m.[39;00mcache_write(block[35;01m=[39;00mb0, write_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mlocal[39m[33m"[39m)
    sch[35;01m.[39;00mreverse_compute_at(block[35;01m=[39;00mb34, loop[35;01m=[39;00ml33, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    b35 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m0[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb35, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l36, l37, l38, l39, l40, l41 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l42 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l40, l41, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v43 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m2[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv43
    )
    b44 [35;01m=[39;00m sch[35;01m.[39;00mcache_read(
        block[35;01m=[39;00mb0, read_buffer_index[35;01m=[39;00m[92m1[39m, storage_scope[35;01m=[39;00m[33m"[39m[33mshared[39m[33m"[39m, consumer_blocks[35;01m=[39;00m[b0]
    )
    sch[35;01m.[39;00mcompute_at(block[35;01m=[39;00mb44, loop[35;01m=[39;00ml28, preserve_unit_loops[35;01m=[39;00m[32;01mTrue[39;00m, index[35;01m=[39;00m[35;01m-[39;00m[92m1[39m)
    l45, l46, l47, l48, l49, l50 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l51 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l49, l50, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    v52 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m1[39m, [92m2[39m, [92m3[39m, [92m4[39m], probs[35;01m=[39;00m[[92m0.25[39m, [92m0.25[39m, [92m0.25[39m, [92m0.25[39m], decision[35;01m=[39;00m[92m0[39m
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m, ann_val[35;01m=[39;00mv52
    )
    l53 [35;01m=[39;00m sch[35;01m.[39;00mfuse(l28, preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_stage[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m0[39m, [92m3[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_order[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m, [92m1[39m, [92m2[39m]
    )
    sch[35;01m.[39;00mannotate(
        block_or_loop[35;01m=[39;00ml53, ann_key[35;01m=[39;00m[33m"[39m[33msoftware_pipeline_async_stages[39m[33m"[39m, ann_val[35;01m=[39;00m[[92m0[39m]
    )
    v54 [35;01m=[39;00m sch[35;01m.[39;00msample_categorical(
        candidates[35;01m=[39;00m[[92m0[39m, [92m16[39m, [92m64[39m, [92m512[39m, [92m1024[39m],
        probs[35;01m=[39;00m[
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
            [92m0.20000000000000001[39m,
        ],
        decision[35;01m=[39;00m[92m3[39m,
    )
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00mb1, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00mv54)
    sch[35;01m.[39;00menter_postproc()
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb35, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l55, l56, l57, l58, l59 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb35)
    l60, l61 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml59, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml61, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb44, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.cooperative_fetch[39m[33m"[39m)
    l62, l63, l64, l65, l66 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb44)
    l67, l68 [35;01m=[39;00m sch[35;01m.[39;00msplit(loop[35;01m=[39;00ml66, factors[35;01m=[39;00m[[32;01mNone[39;00m, [92m128[39m], preserve_unit_iters[35;01m=[39;00m[32;01mTrue[39;00m)
    sch[35;01m.[39;00mbind(loop[35;01m=[39;00ml68, thread_axis[35;01m=[39;00m[33m"[39m[33mthreadIdx.x[39m[33m"[39m)
    b69 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mroot[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    sch[35;01m.[39;00munannotate(block_or_loop[35;01m=[39;00mb69, ann_key[35;01m=[39;00m[33m"[39m[33mmeta_schedule.unroll_explicit[39m[33m"[39m)
    b70, b71, b72, b73 [35;01m=[39;00m sch[35;01m.[39;00mget_child_blocks(b69)
    l74, l75, l76, l77, l78, l79 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb70)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml74, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l80, l81, l82, l83, l84, l85 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb71)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml80, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb72)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml86, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    l96, l97, l98, l99, l100 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(block[35;01m=[39;00mb73)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_auto_unroll_max_step[39m[33m"[39m, ann_val[35;01m=[39;00m[92m512[39m)
    sch[35;01m.[39;00mannotate(block_or_loop[35;01m=[39;00ml96, ann_key[35;01m=[39;00m[33m"[39m[33mpragma_unroll_explicit[39m[33m"[39m, ann_val[35;01m=[39;00m[92m1[39m)
    b101 [35;01m=[39;00m sch[35;01m.[39;00mget_block(name[35;01m=[39;00m[33m"[39m[33mC[39m[33m"[39m, func_name[35;01m=[39;00m[33m"[39m[33mmain[39m[33m"[39m)
    l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 [35;01m=[39;00m sch[35;01m.[39;00mget_loops(
        block[35;01m=[39;00mb101
    )
    b112 [35;01m=[39;00m sch[35;01m.[39;00mdecompose_reduction(block[35;01m=[39;00mb101, loop[35;01m=[39;00ml105)

Final mean time = 0.005283830000000001
