2023-10-23 20:57:34 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 20:57:39 [INFO] LocalBuilder: max_workers = 56
2023-10-23 20:57:40 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:02:50 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:02:50 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:02:54 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:02:54 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:10:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x49459e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x494d768)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x494c888)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x49a9b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1a96b18)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4937e58)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x494e1b8)]: 0 failure(s)
2023-10-23 21:10:17 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:10:18 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:10:18 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 21:10:21 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:10:21 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:10:23 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:15:06 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:15:06 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:15:06 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:15:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:22:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x147a3ff08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1477ba658)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147bd6ee8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14554b648)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x144e50c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147c3d058)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1486130b8)]: 0 failure(s)
2023-10-23 21:22:20 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:22:20 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:22:21 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 21:22:27 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:22:27 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:22:29 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:27:13 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:27:13 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:27:13 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:27:13 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:34:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa0eaff8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xad41a18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14b8529c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa1bcf48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa0bcb08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b996cd8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x71fdd08)]: 0 failure(s)
2023-10-23 21:34:20 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:34:20 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:34:20 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 21:34:28 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:34:28 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:34:29 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:39:17 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:39:17 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:39:17 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:39:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:46:27 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bc1fbd8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x147d889b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a4bd528)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bbd6d68)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ba9d888)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14ba99c98)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f703178)]: 0 failure(s)
2023-10-23 21:46:29 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:46:30 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:46:30 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 21:46:38 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:46:38 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:46:39 [INFO] LocalRunner: max_workers = 1
2023-10-23 21:51:31 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 21:51:31 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 21:51:32 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 21:51:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 21:58:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:58 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:58 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:58 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 13 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:58:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:59:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14ba53d18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x49ffd18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14bc1f9e8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb0aa558)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa5842c8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x147f08bc8)]: 23 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1501f40c8)]: 0 failure(s)
2023-10-23 21:59:00 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 21:59:00 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 21:59:00 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006758 ms
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

2023-10-23 21:59:08 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 21:59:08 [INFO] LocalBuilder: max_workers = 56
2023-10-23 21:59:10 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:04:04 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:04:04 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:04:05 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:04:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:11:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14add7988)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa5437b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4b96e08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14fa8ad38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14ae07488)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4b25438)]: 20 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f6fecc8)]: 0 failure(s)
2023-10-23 22:11:23 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:11:24 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:11:24 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 22:11:31 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:11:31 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:11:33 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:16:17 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:16:17 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:16:18 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:16:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:23:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb067518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14baceb78)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x62fce98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb1b7cf8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa88cc08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a415028)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b56fc48)]: 0 failure(s)
2023-10-23 22:23:35 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:23:35 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:23:35 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 22:23:42 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:23:42 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:23:44 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:28:42 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:28:42 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:28:43 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:28:43 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:35:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b3cc168)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14dc23d18)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d72f708)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a432bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14afc9ab8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14b43b7e8)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a123208)]: 0 failure(s)
2023-10-23 22:35:51 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:35:51 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:35:51 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.004710 ms
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

2023-10-23 22:35:59 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:35:59 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:36:00 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:40:53 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:40:53 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:40:54 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:40:54 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 22:48:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb16f288)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14db616e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a154dd8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14bc51d88)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xaa15408)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9274e88)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x147907a78)]: 0 failure(s)
2023-10-23 22:48:07 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 22:48:08 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 22:48:08 [INFO] [task_scheduler.cc:377] 
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

2023-10-23 22:48:16 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 22:48:16 [INFO] LocalBuilder: max_workers = 56
2023-10-23 22:48:17 [INFO] LocalRunner: max_workers = 1
2023-10-23 22:53:08 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 22:53:08 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 22:53:09 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 22:53:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 23:00:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:30 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14db4d428)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14b568998)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499e27a8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9eef208)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14bc7fec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14974d2b8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1497020f8)]: 0 failure(s)
2023-10-23 23:00:30 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-23 23:00:31 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-23 23:00:31 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

Time cost of MyModule after tuning: 0.006861 ms
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

Final Mean Time = 0.00533503
