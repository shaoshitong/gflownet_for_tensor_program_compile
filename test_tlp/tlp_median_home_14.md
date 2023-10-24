2023-10-17 17:09:07 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 17:09:14 [INFO] LocalBuilder: max_workers = 56
2023-10-17 17:09:16 [INFO] LocalRunner: max_workers = 1
2023-10-17 17:14:42 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 17:14:42 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 17:14:46 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 17:14:46 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 17:22:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5c91db8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d10718)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5bf8978)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5ca3bb8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5cb43b8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5bf7308)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0b208)]: 0 failure(s)
2023-10-17 17:22:22 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 17:22:23 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 17:22:23 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 17:22:26 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 17:22:26 [INFO] LocalBuilder: max_workers = 56
2023-10-17 17:22:27 [INFO] LocalRunner: max_workers = 1
2023-10-17 17:27:25 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 17:27:25 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 17:27:26 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 17:27:26 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 17:34:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:34:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:34:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:34:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:35:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:35:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:35:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x13302a518)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x145a70a98)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x144d6e698)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x145725068)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x145726138)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14519aa88)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1451b5ab8)]: 0 failure(s)
2023-10-17 17:35:00 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 17:35:01 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 17:35:01 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 17:35:06 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 17:35:06 [INFO] LocalBuilder: max_workers = 56
2023-10-17 17:35:08 [INFO] LocalRunner: max_workers = 1
2023-10-17 17:39:52 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 17:39:52 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 17:39:53 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 17:39:53 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 17:47:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 16 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbab4e48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xc49fe08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb59da58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc4364f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3ad4d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x148f5a4e8)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5bf8808)]: 0 failure(s)
2023-10-17 17:47:34 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 17:47:35 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 17:47:35 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 17:47:42 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 17:47:42 [INFO] LocalBuilder: max_workers = 56
2023-10-17 17:47:44 [INFO] LocalRunner: max_workers = 1
2023-10-17 17:52:39 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 17:52:39 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 17:52:39 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 17:52:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:00:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da60ad8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d828518)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147871de8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14da599f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x148ea2c78)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4c8a08)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x6af9f78)]: 0 failure(s)
2023-10-17 18:00:22 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:00:23 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:00:23 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:00:30 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:00:30 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:00:32 [INFO] LocalRunner: max_workers = 1
2023-10-17 18:05:35 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 18:05:35 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 18:05:36 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 18:05:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:12:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:12:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xc3ccf08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x146662248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14be8ba38)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14506ae48)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc14a388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xc4de6a8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14be88048)]: 0 failure(s)
2023-10-17 18:13:01 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:13:01 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:13:01 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:13:08 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:13:08 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:13:11 [INFO] LocalRunner: max_workers = 1
2023-10-17 18:17:57 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 18:17:57 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 18:17:58 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 18:17:58 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:24:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 10 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:24 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:24 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 24 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:24 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 24 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 27 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:25 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbfdb568)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148d06508)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14e5eb3d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x148fdb7f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xb3a8738)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14651ef68)]: 27 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14b9412d8)]: 0 failure(s)
2023-10-17 18:24:25 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:24:25 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:24:25 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:24:31 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:24:31 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:24:33 [INFO] LocalRunner: max_workers = 1
2023-10-17 18:29:15 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 18:29:15 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 18:29:16 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 18:29:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:35:37 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 1 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 1 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb4356a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14ba2b728)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x147aa7538)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb237508)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xc0efc38)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14cdb08b8)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1473d93d8)]: 0 failure(s)
2023-10-17 18:35:40 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:35:40 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:35:40 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:35:46 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:35:46 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:35:48 [INFO] LocalRunner: max_workers = 1
2023-10-17 18:40:35 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 18:40:35 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 18:40:35 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 18:40:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:46:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 7 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 18 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14bdbc5a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e701898)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1451bb068)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xb456128)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c8fcc58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14e838838)]: 21 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1471431f8)]: 0 failure(s)
2023-10-17 18:47:01 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:47:02 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:47:02 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:47:08 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:47:08 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:47:09 [INFO] LocalRunner: max_workers = 1
2023-10-17 18:52:04 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 18:52:04 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 18:52:05 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 18:52:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 18:58:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:42 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 15 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:42 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:42 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 19 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:43 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14da50d68)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e694de8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14d5e2258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xc152538)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d5eff58)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x146ae7e58)]: 22 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14dac8208)]: 0 failure(s)
2023-10-17 18:58:43 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 18:58:43 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 18:58:43 [INFO] [task_scheduler.cc:377] 
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

2023-10-17 18:58:49 [INFO] Logging directory: ./tune_tmp/logs
2023-10-17 18:58:49 [INFO] LocalBuilder: max_workers = 56
2023-10-17 18:58:50 [INFO] LocalRunner: max_workers = 1
2023-10-17 19:03:48 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-17 19:03:48 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-17 19:03:49 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-17 19:03:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-17 19:09:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 2 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a7d4998)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14e665d28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1481f1648)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14cd9b268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14d8f7a08)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x1481e0488)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x148facc88)]: 0 failure(s)
2023-10-17 19:09:23 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:512] Cannot sample enough initial population, evolutionary search failed.
2023-10-17 19:09:23 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-17 19:09:23 [INFO] [task_scheduler.cc:377] 
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

Final Mean Time = 0.004976630000000001
