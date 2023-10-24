
(4.2004+4.06544+4.14425+4.13247+4.29683+4.19097+4.30779)/7 = 4.19116

2023-10-01 20:50:38 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 20:50:41 [INFO] LocalBuilder: max_workers = 56
2023-10-01 20:50:43 [INFO] LocalRunner: max_workers = 1
2023-10-01 20:55:19 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 20:55:19 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 20:55:22 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 20:55:22 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 20:59:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1597 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5e25ae8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d1a538)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5d19658)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5e20a28)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5dd8ea8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5e232a8)]: 387 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0c138)]: 0 failure(s)
2023-10-01 20:59:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x5e25ae8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5d1a538)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5d19658)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x5e20a28)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5dd8ea8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5e232a8)]: 387 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x5d0c138)]: 0 failure(s)
2023-10-01 20:59:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 64 candidate(s)
2023-10-01 21:15:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3236 3.3037 3.3022 3.3365 3.2958 3.2342 3.2923 3.2916 3.3014 3.2848 3.2938 3.3258

[17 : 32]:	3.2481 3.2572 3.2958 3.2332 3.2342 3.2257 3.2716 3.2761 3.2893 3.2624 3.2267 3.2257 3.2636 3.2414 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2332 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2561 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2383 3.2593 3.1828 3.1204 2.4764 3.1992 3.2093 3.2275 3.2238 3.2039 3.1555 3.1611 3.2296

2023-10-01 21:15:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 21:15:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 21:15:30 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 21:16:02 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 21:16:37 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 21:16:37 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       998.5491 |       4.2004 |                4.2004 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.2004

2023-10-01 21:16:37 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 21:16:37 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       998.5491 |       4.2004 |                4.2004 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.2004

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

2023-10-01 21:16:41 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 21:16:41 [INFO] LocalBuilder: max_workers = 56
2023-10-01 21:16:42 [INFO] LocalRunner: max_workers = 1
2023-10-01 21:21:15 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 21:21:15 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 21:21:16 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 21:21:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 21:25:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1661 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149eed138)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14f6308a8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xc541d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14ca1cec8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149ed6158)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xad36788)]: 353 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xb8b12b8)]: 0 failure(s)
2023-10-01 21:25:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149eed138)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14f6308a8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xc541d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14ca1cec8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149ed6158)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xad36788)]: 353 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xb8b12b8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149eed138)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14f6308a8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xc541d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14ca1cec8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149ed6158)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xad36788)]: 693 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xb8b12b8)]: 0 failure(s)
2023-10-01 21:26:24 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x149eed138)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14f6308a8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xc541d68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14ca1cec8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x149ed6158)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xad36788)]: 693 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xb8b12b8)]: 0 failure(s)
2023-10-01 21:26:25 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 81 candidate(s)
2023-10-01 21:43:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3236 3.3037 3.3022 3.3365 3.3258 3.2342 3.2923 3.2916 3.3014 3.2848 3.2938 3.2958

[17 : 32]:	3.2481 3.2572 3.2958 3.2332 3.2342 3.2257 3.2716 3.2761 3.2893 3.2624 3.2267 3.2414 3.2636 3.2383 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2332 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2561 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2257 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2296

2023-10-01 21:43:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 21:43:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 21:43:11 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 21:43:33 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 21:44:15 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 21:44:15 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1031.6978 |       4.0654 |                4.0654 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.06544

2023-10-01 21:44:15 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 21:44:15 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1031.6978 |       4.0654 |                4.0654 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.06544

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

2023-10-01 21:44:22 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 21:44:22 [INFO] LocalBuilder: max_workers = 56
2023-10-01 21:44:23 [INFO] LocalRunner: max_workers = 1
2023-10-01 21:48:59 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 21:48:59 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 21:49:00 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 21:49:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 21:53:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1725 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14999d908)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1488e2278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149291f68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1499af488)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x80e9bd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a5c44c8)]: 283 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35de8e78)]: 0 failure(s)
2023-10-01 21:53:45 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14999d908)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1488e2278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149291f68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1499af488)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x80e9bd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a5c44c8)]: 283 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35de8e78)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14999d908)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1488e2278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149291f68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1499af488)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x80e9bd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a5c44c8)]: 562 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35de8e78)]: 0 failure(s)
2023-10-01 21:54:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14999d908)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x1488e2278)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x149291f68)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1499af488)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x80e9bd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14a5c44c8)]: 562 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35de8e78)]: 0 failure(s)
2023-10-01 21:54:13 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 84 candidate(s)
2023-10-01 22:11:55 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3657 3.339 3.3258 3.3384 3.3236 3.3037 3.3022 3.3365 3.3258 3.2342 3.2923 3.2916 3.3014 3.2848 3.2938 3.2958

[17 : 32]:	3.2481 3.2572 3.2958 3.2332 3.2342 3.2257 3.2716 3.2761 3.2893 3.2624 3.2267 3.2414 3.2636 3.2383 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2332 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2561 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2257 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2296

2023-10-01 22:11:55 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 22:11:55 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 22:11:56 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 22:12:18 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 22:12:53 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 22:12:53 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1012.0772 |       4.1443 |                4.1443 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.14425

2023-10-01 22:12:53 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 22:12:53 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1012.0772 |       4.1443 |                4.1443 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.14425

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

2023-10-01 22:12:59 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 22:12:59 [INFO] LocalBuilder: max_workers = 56
2023-10-01 22:13:01 [INFO] LocalRunner: max_workers = 1
2023-10-01 22:17:39 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 22:17:39 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 22:17:40 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 22:17:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 22:22:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1789 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a5cd5e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148f64f48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5e2fba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a8a7948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xad3a548)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149375ad8)]: 225 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f7b6a21c508)]: 0 failure(s)
2023-10-01 22:22:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a5cd5e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148f64f48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5e2fba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a8a7948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xad3a548)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149375ad8)]: 225 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f7b6a21c508)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a5cd5e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148f64f48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5e2fba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a8a7948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xad3a548)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149375ad8)]: 453 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f7b6a21c508)]: 0 failure(s)
2023-10-01 22:22:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14a5cd5e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x148f64f48)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x5e2fba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14a8a7948)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xad3a548)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149375ad8)]: 453 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f7b6a21c508)]: 0 failure(s)
2023-10-01 22:22:51 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 65 candidate(s)
2023-10-01 22:39:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3236 3.3037 3.3022 3.3365 3.3258 3.2342 3.2923 3.2916 3.3014 3.2848 3.2938 3.2958

[17 : 32]:	3.2481 3.2572 3.2958 3.2332 3.2342 3.2257 3.2716 3.2761 3.2893 3.2624 3.2267 3.2414 3.2636 3.2383 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2332 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2561 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2257 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2296

2023-10-01 22:39:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 22:39:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 22:39:32 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 22:39:54 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 22:40:29 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 22:40:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1014.9626 |       4.1325 |                4.1325 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.13247

2023-10-01 22:40:29 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 22:40:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1014.9626 |       4.1325 |                4.1325 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.13247

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

2023-10-01 22:40:36 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 22:40:36 [INFO] LocalBuilder: max_workers = 56
2023-10-01 22:40:38 [INFO] LocalRunner: max_workers = 1
2023-10-01 22:45:12 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 22:45:12 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 22:45:12 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 22:45:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 22:49:49 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1853 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 174 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
2023-10-01 22:50:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 174 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 342 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
2023-10-01 22:50:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 342 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 512 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
2023-10-01 22:50:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xbdeb7e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x149ad88e8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x1499d8348)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x87af8d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x1499143e8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x149ae1328)]: 512 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35fedcc8)]: 0 failure(s)
2023-10-01 22:50:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 73 candidate(s)
2023-10-01 23:05:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3014 3.3258 3.3236 3.3365 3.3037 3.2923 3.2716 3.2893 3.2916 3.3022 3.2938 3.2958

[17 : 32]:	3.2848 3.2572 3.2958 3.2257 3.2332 3.2342 3.2561 3.2296 3.2481 3.2624 3.2267 3.2342 3.2636 3.2761 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2414 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2383 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2332 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2257

2023-10-01 23:05:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 23:05:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 23:05:37 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 23:06:00 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 23:06:36 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 23:06:36 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       976.1400 |       4.2968 |                4.2968 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.29683

2023-10-01 23:06:36 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 23:06:36 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       976.1400 |       4.2968 |                4.2968 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.29683

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

2023-10-01 23:06:43 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 23:06:43 [INFO] LocalBuilder: max_workers = 56
2023-10-01 23:06:45 [INFO] LocalRunner: max_workers = 1
2023-10-01 23:11:27 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 23:11:27 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 23:11:27 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 23:11:27 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 23:16:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1917 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 110 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
2023-10-01 23:16:24 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 110 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 229 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
2023-10-01 23:16:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 229 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 345 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
2023-10-01 23:16:45 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 345 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 462 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
2023-10-01 23:16:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7f6f33d79f48)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f35f22a28)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x150306498)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b8935c6d8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14a6b9988)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f3920b838)]: 462 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f30413098)]: 0 failure(s)
2023-10-01 23:16:57 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 62 candidate(s)
2023-10-01 23:31:17 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3014 3.3258 3.3236 3.3365 3.3037 3.2923 3.2716 3.2893 3.2916 3.3022 3.2938 3.2958

[17 : 32]:	3.2848 3.2572 3.2958 3.2257 3.2332 3.2342 3.2561 3.2296 3.2481 3.2624 3.2267 3.2342 3.2636 3.2761 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2414 3.2298 3.2315 3.2915 3.1789 3.211 3.2018 3.204 3.2061 3.2068 3.2018 3.2383 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2332 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2257

2023-10-01 23:31:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 23:31:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 23:31:18 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 23:31:42 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-01 23:32:17 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-01 23:32:17 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1000.7962 |       4.1910 |                4.1910 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.19097

2023-10-01 23:32:17 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-01 23:32:17 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1000.7962 |       4.1910 |                4.1910 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.19097

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

2023-10-01 23:32:24 [INFO] Logging directory: ./tune_tmp/logs
2023-10-01 23:32:24 [INFO] LocalBuilder: max_workers = 56
2023-10-01 23:32:25 [INFO] LocalRunner: max_workers = 1
2023-10-01 23:37:03 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-01 23:37:03 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-01 23:37:04 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-01 23:37:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-01 23:42:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1981 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 59 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 59 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 121 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 121 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 178 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 178 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 237 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 237 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 296 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 296 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 353 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:34 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 353 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 411 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14b1c0348)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f3681ca88)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f921c98)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xbe16598)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x7112568)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x7f6f2f6ff9d8)]: 411 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14f63acd8)]: 0 failure(s)
2023-10-01 23:42:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 58 candidate(s)
2023-10-01 23:59:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	3.3556 3.339 3.3258 3.3384 3.3014 3.3258 3.3236 3.3365 3.3037 3.2923 3.2716 3.2893 3.2916 3.3022 3.2938 3.2958

[17 : 32]:	3.2848 3.2572 3.2958 3.2257 3.2332 3.2342 3.2561 3.2296 3.2481 3.2624 3.2267 3.2342 3.2636 3.2761 3.1868 3.2723

[33 : 48]:	3.254 3.2147 3.238 3.2414 3.2298 3.2315 3.2915 3.1789 3.2018 3.2018 3.204 3.2061 3.2068 3.211 3.2383 3.2079

[49 : 64]:	3.2122 3.2074 3.2133 3.2332 3.2593 3.1828 3.1204 2.4764 3.2093 3.2275 3.1992 3.2238 3.2039 3.1555 3.1611 3.2257

2023-10-01 23:59:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-10-01 23:59:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-10-01 23:59:29 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-10-01 23:59:58 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-10-02 00:00:37 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-10-02 00:00:37 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       973.6548 |       4.3078 |                4.3078 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.30779

2023-10-02 00:00:37 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-02 00:00:37 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       973.6548 |       4.3078 |                4.3078 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.30779

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

2023-10-02 00:00:44 [INFO] Logging directory: ./tune_tmp/logs
2023-10-02 00:00:44 [INFO] LocalBuilder: max_workers = 56
2023-10-02 00:00:46 [INFO] LocalRunner: max_workers = 1
2023-10-02 00:05:16 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-02 00:05:16 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-02 00:05:16 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-02 00:05:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-02 00:10:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 12 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x14f8e70a8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7f6f352f7b08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14c0603f8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x1491fa398)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f0968a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c17a368)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x14a76a928)]: 0 failure(s)
2023-10-02 00:10:20 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:510] Cannot sample enough initial population, evolutionary search failed.
2023-10-02 00:10:21 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-02 00:10:21 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

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

2023-10-02 00:10:27 [INFO] Logging directory: ./tune_tmp/logs
2023-10-02 00:10:27 [INFO] LocalBuilder: max_workers = 56
2023-10-02 00:10:29 [INFO] LocalRunner: max_workers = 1
2023-10-02 00:15:11 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-02 00:15:11 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-02 00:15:11 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-02 00:15:11 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-02 00:20:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:18 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 4 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 5 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 8 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:20 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xb417618)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x14d497378)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14f1f74c8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x7f7b6a21c758)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14c229fd8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x14c00f4a8)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7f6f35e378d8)]: 0 failure(s)
2023-10-02 00:20:20 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:510] Cannot sample enough initial population, evolutionary search failed.
2023-10-02 00:20:20 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-02 00:20:20 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

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

2023-10-02 00:20:26 [INFO] Logging directory: ./tune_tmp/logs
2023-10-02 00:20:26 [INFO] LocalBuilder: max_workers = 56
2023-10-02 00:20:28 [INFO] LocalRunner: max_workers = 1
2023-10-02 00:25:01 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-02 00:25:01 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-02 00:25:02 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-02 00:25:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-10-02 00:30:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 2045 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 3 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 6 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:08 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 9 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 11 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 14 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x81780e8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x5f24118)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x14a887a08)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x14b351048)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x14f3915a8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x79c5288)]: 17 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xc31e1c8)]: 0 failure(s)
2023-10-02 00:30:09 [WARNING] [tvm.meta_schedule.search_strategy.gflownet_search:510] Cannot sample enough initial population, evolutionary search failed.
2023-10-02 00:30:10 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-10-02 00:30:10 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |            N/A |          N/A |                   N/A |      0 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

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

