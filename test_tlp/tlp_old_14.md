2023-10-23 21:02:03 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:02:10 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:02:11 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:07:37 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:07:37 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:07:40 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:07:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:14:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:52 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:52 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:52 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x6307f88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x63347f8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x6328e38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x6234088)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x625a5b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6226728)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6235208)]: 0 failure(s)
2023-10-23 21:14:52 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:14:53 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:14:53 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.004608 ms
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

2023-10-23 21:14:57 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:14:57 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:14:59 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:19:42 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:19:42 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:19:42 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:19:42 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:26:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149c81a88)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14a029d88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a02b848)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149e2df88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149fc3c58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14996c018)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14af6a958)]: 0 failure(s)
2023-10-23 21:26:51 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:26:51 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:26:52 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005632 ms
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

2023-10-23 21:27:01 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:27:01 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:27:03 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:31:41 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:31:41 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:31:42 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:31:42 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:38:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbb2cf78)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xca8ed68)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d877d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbb2af48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xbe89b08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d9fa588)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xbf33818)]: 0 failure(s)
2023-10-23 21:38:51 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:38:52 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:38:52 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.004608 ms
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

2023-10-23 21:38:59 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:38:59 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:39:01 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:43:50 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:43:50 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:43:50 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:43:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:51:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:03 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc9f3058)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15231bb18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d2aa7a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb634878)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7418398)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14aaad948)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1499c47d8)]: 0 failure(s)
2023-10-23 21:51:03 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:51:04 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:51:04 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006349 ms
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

2023-10-23 21:51:13 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:51:13 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:51:16 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:56:15 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:56:15 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:56:16 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:56:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:03:43 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:44 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:44 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:44 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:45 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:45 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:46 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:46 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbc3afd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dd05a58)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14cd20a48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a11e598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc44d938)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14dd07608)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14db65768)]: 0 failure(s)
2023-10-23 22:03:46 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:03:48 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:03:48 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005632 ms
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

2023-10-23 22:03:58 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:03:58 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:04:00 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:08:51 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:08:51 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:08:51 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:08:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:16:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 24 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 24 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 26 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 26 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 29 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 29 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 32 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da53b08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1530edee8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14dbc1b58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x149ba0b78)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dc94018)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x6425cf8)]: 32 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6c25028)]: 0 failure(s)
2023-10-23 22:16:10 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:16:11 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:16:11 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005427 ms
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

2023-10-23 22:16:18 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:16:18 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:16:20 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:21:09 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:21:09 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:21:10 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:21:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:28:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb977118)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x15267d078)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149cb8508)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc0ca928)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb89fdf8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1526f5a98)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152730188)]: 0 failure(s)
2023-10-23 22:28:22 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:28:22 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:28:22 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005530 ms
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

2023-10-23 22:28:32 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:28:32 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:28:34 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:33:21 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:33:21 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:33:21 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:33:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:40:27 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14be61628)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x152795a48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x151229958)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d1da9e8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14dadbc98)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d90ea28)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x152696fd8)]: 0 failure(s)
2023-10-23 22:40:30 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:40:30 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:40:30 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005632 ms
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

2023-10-23 22:40:40 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:40:40 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:40:42 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:45:29 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:45:29 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:45:29 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:45:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:52:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:37 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:37 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc6a0898)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150790868)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d7c06c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbf344c8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14cd7f658)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14d55ecb8)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xac03b98)]: 0 failure(s)
2023-10-23 22:52:37 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:52:37 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:52:38 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.005120 ms
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

2023-10-23 22:52:46 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:52:46 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:52:48 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:57:36 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:57:36 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:57:37 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:57:37 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 23:04:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14c655d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x150af47e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x153675f48)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14d84ed18)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x97cb708)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x94510b8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1506da528)]: 0 failure(s)
2023-10-23 23:04:02 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 23:04:03 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 23:04:03 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006451 ms
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

Final Mean Time = 0.0054988400000000005
