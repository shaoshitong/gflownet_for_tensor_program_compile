# overview
(4.20915+4.17598+4.49134+4.2011+4.15169+4.11882+4.33921+4.10428+4.19142+4.42763)/10 = 4.241062

2023-09-29 15:46:40 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 15:46:45 [INFO] LocalBuilder: max_workers = 56
2023-09-29 15:46:47 [INFO] LocalRunner: max_workers = 1
2023-09-29 15:51:56 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 15:51:56 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 15:51:59 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 15:51:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 15:53:23 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 575 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x4de7368)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x4e3e3c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4d22ed8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4d8d2b8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x4db2328)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4d2f5e8)]: 1292 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x4d3c6d8)]: 0 failure(s)
2023-09-29 15:55:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x4de7368)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x4e3e3c8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4d22ed8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4d8d2b8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x4db2328)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4d2f5e8)]: 1292 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x4d3c6d8)]: 0 failure(s)
2023-09-29 15:55:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 181 candidate(s)
2023-09-29 16:09:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9962 0.9908 0.9789 0.9803 0.951 0.925 0.7981 0.97 0.9648 0.8374 0.8425 0.4759 0.7987 0.61 0.7732 0.8313

[17 : 32]:	0.7026 0.702 0.4612 0.5044 0.6477 0.767 0.318 0.3216 0.3922 0.4657 0.3718 0.5787 0.3838 0.5561 0.6619 0.8221

[33 : 48]:	0.5386 0.4505 0.3183 0.2316 0.4106 0.3381 0.3789 0.2749 0.454 0.374 0.61 0.4102 0.0126 0.0696 0.2086 0.1112

[49 : 64]:	0.3087 0.159 0.1786 0.1842 0.3132 0.0487 0.2175 0.275 0.4401 0.3188 0.3328 0.2926 0.1187 0.234 0.1161 0.663

2023-09-29 16:09:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 16:09:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 16:09:32 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 16:09:45 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 16:10:22 [DEBUG] XGB iter   0: tr-p-rmse: 0.314597	tr-a-peak@32: 0.893879	tr-rmse: 0.574688	tr-rmse: 0.574688
2023-09-29 16:10:22 [DEBUG] XGB iter  25: tr-p-rmse: 0.022207	tr-a-peak@32: 0.998319	tr-rmse: 0.628964	tr-rmse: 0.628964
2023-09-29 16:10:22 [DEBUG] XGB iter  50: tr-p-rmse: 0.022207	tr-a-peak@32: 0.998319	tr-rmse: 0.628964	tr-rmse: 0.628964
2023-09-29 16:10:22 [DEBUG] XGB stopped. Best iteration: [17] tr-p-rmse:0.02221	tr-a-peak@32:0.99832	tr-rmse:0.62896	tr-rmse:0.62896 
2023-09-29 16:10:22 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 16:10:22 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       996.4733 |       4.2091 |                4.2091 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.20915

2023-09-29 16:10:22 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 16:10:22 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       996.4733 |       4.2091 |                4.2091 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.20915

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

2023-09-29 16:10:25 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 16:10:25 [INFO] LocalBuilder: max_workers = 56
2023-09-29 16:10:26 [INFO] LocalRunner: max_workers = 1
2023-09-29 16:14:59 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 16:14:59 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 16:15:00 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 16:15:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 16:17:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 639 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x9432b18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa4daca8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x84d4828)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa11f778)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8f1748)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xabebef8)]: 1242 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xa0d3da8)]: 0 failure(s)
2023-09-29 16:19:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x9432b18)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa4daca8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x84d4828)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa11f778)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8f1748)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xabebef8)]: 1242 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xa0d3da8)]: 0 failure(s)
2023-09-29 16:19:40 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 167 candidate(s)
2023-09-29 16:31:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9997 0.973 0.9387 0.9311 0.8635 0.7973 0.8127 0.9011 0.7618 0.6368 0.7514 0.7491 0.7911 0.6681 0.7667 0.8961

[17 : 32]:	0.6799 0.7353 0.4304 0.5882 0.4182 0.4955 0.652 0.5173 0.7364 0.6632 0.6831 0.5181 0.5398 0.6868 0.5671 0.8441

[33 : 48]:	0.106 0.1778 0.4941 0.4929 0.6521 0.1229 0.1618 0.0286 0.3395 0.0715 0.1048 0.16 0.2315 0.1801 0.3655 0.2249

[49 : 64]:	0.2598 0.3939 0.4327 0.2771 0.3151 0.6472 0.6693 0.1233 0.5089 0.2897 0.531 0.0959 0.3964 0.0659 0.0668 0.378

2023-09-29 16:31:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 16:31:39 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 16:31:40 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 16:31:53 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 16:32:29 [DEBUG] XGB iter   0: tr-p-rmse: 0.327947	tr-a-peak@32: 0.946352	tr-rmse: 0.587031	tr-rmse: 0.587031
2023-09-29 16:32:29 [DEBUG] XGB iter  25: tr-p-rmse: 0.027914	tr-a-peak@32: 1.000000	tr-rmse: 0.641808	tr-rmse: 0.641808
2023-09-29 16:32:29 [DEBUG] XGB iter  50: tr-p-rmse: 0.027914	tr-a-peak@32: 1.000000	tr-rmse: 0.641808	tr-rmse: 0.641808
2023-09-29 16:32:29 [DEBUG] XGB stopped. Best iteration: [11] tr-p-rmse:0.02791	tr-a-peak@32:1.00000	tr-rmse:0.64181	tr-rmse:0.64181 
2023-09-29 16:32:29 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 16:32:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1004.3878 |       4.1760 |                4.1760 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.17598

2023-09-29 16:32:29 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 16:32:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1004.3878 |       4.1760 |                4.1760 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.17598

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

2023-09-29 16:32:35 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 16:32:35 [INFO] LocalBuilder: max_workers = 56
2023-09-29 16:32:36 [INFO] LocalRunner: max_workers = 1
2023-09-29 16:37:12 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 16:37:12 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 16:37:12 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 16:37:12 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 16:39:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 767 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa50a6b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7cc4548)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xa11b458)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa50a718)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae96388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9bf66c8)]: 1131 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x8346108)]: 0 failure(s)
2023-09-29 16:41:07 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa50a6b8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7cc4548)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xa11b458)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa50a718)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xae96388)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9bf66c8)]: 1131 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x8346108)]: 0 failure(s)
2023-09-29 16:41:09 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 150 candidate(s)
2023-09-29 16:56:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9999 0.9826 0.8827 0.9629 0.8183 0.8687 0.8541 0.931 0.8763 0.7913 0.6443 0.7921 0.7277 0.7729 0.8163 0.805

[17 : 32]:	0.6232 0.6585 0.8624 0.6888 0.6122 0.4225 0.6362 0.692 0.7441 0.5158 0.6629 0.4877 0.7313 0.4848 0.3967 0.7454

[33 : 48]:	0.1537 0.3803 0.344 0.0024 0.4697 0.5873 0.6827 0.2574 0.6435 0.2772 0.1154 0.1976 0.4203 0.2902 0.3579 0.1355

[49 : 64]:	0.6213 0.2728 0.0838 0.0493 0.3555 0.1977 0.2041 0.0746 0.2741 0.1562 0.3461 0.1934 0.2648 0.3471 0.0364 0.2081

2023-09-29 16:56:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 16:56:29 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 16:56:30 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 16:56:51 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 16:57:27 [DEBUG] XGB iter   0: tr-p-rmse: 0.326743	tr-a-peak@32: 0.912813	tr-rmse: 0.613405	tr-rmse: 0.613405
2023-09-29 16:57:27 [DEBUG] XGB iter  25: tr-p-rmse: 0.025869	tr-a-peak@32: 0.999115	tr-rmse: 0.666684	tr-rmse: 0.666684
2023-09-29 16:57:28 [DEBUG] XGB iter  50: tr-p-rmse: 0.025869	tr-a-peak@32: 0.999115	tr-rmse: 0.666684	tr-rmse: 0.666684
2023-09-29 16:57:28 [DEBUG] XGB stopped. Best iteration: [10] tr-p-rmse:0.02587	tr-a-peak@32:0.99911	tr-rmse:0.66668	tr-rmse:0.66668 
2023-09-29 16:57:28 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 16:57:28 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       933.8642 |       4.4913 |                4.4913 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.49134

2023-09-29 16:57:28 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 16:57:28 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       933.8642 |       4.4913 |                4.4913 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.49134

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

2023-09-29 16:57:34 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 16:57:34 [INFO] LocalBuilder: max_workers = 56
2023-09-29 16:57:35 [INFO] LocalRunner: max_workers = 1
2023-09-29 17:02:06 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 17:02:06 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 17:02:06 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 17:02:06 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 17:04:19 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 895 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa5da478)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7d86818)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x9f2f2d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xacd8b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8e9cc8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xaf22338)]: 1003 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1086e288)]: 0 failure(s)
2023-09-29 17:05:58 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xa5da478)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x7d86818)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x9f2f2d8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xacd8b38)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa8e9cc8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xaf22338)]: 1003 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x1086e288)]: 0 failure(s)
2023-09-29 17:05:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 150 candidate(s)
2023-09-29 17:21:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9989 0.9747 0.9646 0.9469 0.9295 0.8852 0.9271 0.9106 0.8806 0.8419 0.8093 0.7552 0.8061 0.8063 0.8656 0.7485

[17 : 32]:	0.3489 0.6569 0.7595 0.7944 0.5033 0.4433 0.6346 0.3549 0.6666 0.6323 0.7868 0.6617 0.634 0.6223 0.7028 0.7153

[33 : 48]:	0.2393 0.0481 0.0648 0.1164 0.337 0.3748 0.7467 0.0157 0.2735 0.166 0.1649 0.3138 0.2407 0.1889 0.4489 0.0626

[49 : 64]:	0.3546 0.2292 0.3321 0.0648 0.2967 0.0999 0.3619 0.1154 0.2054 0.4356 0.0189 0.1335 0.1708 0.4868 0.6442 0.3972

2023-09-29 17:21:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 17:21:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 17:21:32 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 17:21:45 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 17:22:21 [DEBUG] XGB iter   0: tr-p-rmse: 0.325978	tr-a-peak@32: 0.891267	tr-rmse: 0.588365	tr-rmse: 0.588365
2023-09-29 17:22:22 [DEBUG] XGB iter  25: tr-p-rmse: 0.023658	tr-a-peak@32: 1.000000	tr-rmse: 0.642877	tr-rmse: 0.642877
2023-09-29 17:22:22 [DEBUG] XGB iter  50: tr-p-rmse: 0.023658	tr-a-peak@32: 1.000000	tr-rmse: 0.642877	tr-rmse: 0.642877
2023-09-29 17:22:22 [DEBUG] XGB stopped. Best iteration: [11] tr-p-rmse:0.02366	tr-a-peak@32:1.00000	tr-rmse:0.64288	tr-rmse:0.64288 
2023-09-29 17:22:22 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 17:22:22 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       998.3812 |       4.2011 |                4.2011 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.2011

2023-09-29 17:22:22 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 17:22:22 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       998.3812 |       4.2011 |                4.2011 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.2011

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

2023-09-29 17:22:28 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 17:22:28 [INFO] LocalBuilder: max_workers = 56
2023-09-29 17:22:29 [INFO] LocalRunner: max_workers = 1
2023-09-29 17:26:58 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 17:26:58 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 17:26:59 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 17:26:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 17:29:31 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1023 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xaa7d878)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x12dca248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xe561318)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9fb3268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa070418)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4eba9c8)]: 898 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7a258c8)]: 0 failure(s)
2023-09-29 17:31:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xaa7d878)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x12dca248)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xe561318)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x9fb3268)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa070418)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x4eba9c8)]: 898 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x7a258c8)]: 0 failure(s)
2023-09-29 17:31:02 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 127 candidate(s)
2023-09-29 17:47:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9997 0.9887 0.9818 0.9109 0.8607 0.9598 0.9562 0.8796 0.7407 0.8073 0.7188 0.925 0.935 0.5674 0.658 0.6874

[17 : 32]:	0.5578 0.5759 0.6065 0.7849 0.7263 0.34 0.5039 0.545 0.5769 0.7967 0.5822 0.3652 0.3809 0.2285 0.6445 0.6453

[33 : 48]:	0.4445 0.5553 0.4004 0.0554 0.2261 0.0681 0.1874 0.173 0.2784 0.4131 0.0181 0.1194 0.2138 0.3913 0.4499 0.0778

[49 : 64]:	0.5125 0.4442 0.4282 0.4371 0.0905 0.5495 0.4018 0.124 0.2634 0.2538 0.1132 0.1335 0.0493 0.0949 0.1353 0.3743

2023-09-29 17:47:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 17:47:32 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 17:47:33 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 17:47:46 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 17:48:24 [DEBUG] XGB iter   0: tr-p-rmse: 0.309252	tr-a-peak@32: 0.917139	tr-rmse: 0.578427	tr-rmse: 0.578427
2023-09-29 17:48:24 [DEBUG] XGB iter  25: tr-p-rmse: 0.024046	tr-a-peak@32: 1.000000	tr-rmse: 0.632536	tr-rmse: 0.632536
2023-09-29 17:48:24 [DEBUG] XGB iter  50: tr-p-rmse: 0.024046	tr-a-peak@32: 1.000000	tr-rmse: 0.632536	tr-rmse: 0.632536
2023-09-29 17:48:24 [DEBUG] XGB stopped. Best iteration: [24] tr-p-rmse:0.02405	tr-a-peak@32:1.00000	tr-rmse:0.63254	tr-rmse:0.63254 
2023-09-29 17:48:24 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 17:48:24 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1010.2638 |       4.1517 |                4.1517 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.15169

2023-09-29 17:48:24 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 17:48:24 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1010.2638 |       4.1517 |                4.1517 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.15169

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

2023-09-29 17:48:30 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 17:48:30 [INFO] LocalBuilder: max_workers = 56
2023-09-29 17:48:32 [INFO] LocalRunner: max_workers = 1
2023-09-29 17:52:59 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 17:52:59 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 17:53:00 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 17:53:00 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 17:55:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1150 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xada73c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa66dbf8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x12cc5158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa4f7c58)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5218088)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x61652e8)]: 782 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x104ffd78)]: 0 failure(s)
2023-09-29 17:57:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xada73c8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xa66dbf8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x12cc5158)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa4f7c58)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x5218088)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x61652e8)]: 782 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x104ffd78)]: 0 failure(s)
2023-09-29 17:57:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 116 candidate(s)
2023-09-29 18:10:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9996 0.9776 0.9539 0.9751 0.9329 0.9172 0.8356 0.9742 0.8629 0.9197 0.9324 0.8921 0.8933 0.7515 0.7322 0.7855

[17 : 32]:	0.7046 0.521 0.6332 0.3661 0.8626 0.3747 0.7737 0.8242 0.6487 0.5103 0.803 0.414 0.1986 0.5955 0.6743 0.5893

[33 : 48]:	0.1415 0.1274 0.4954 0.1755 0.3191 0.3149 0.0262 0.2141 0.2363 0.122 0.7041 0.3113 0.132 0.3746 0.0027 0.5225

[49 : 64]:	0.7077 0.0077 0.6463 0.4087 0.1895 0.2929 0.4711 0.1154 0.3589 0.1407 0.0714 0.0365 0.3728 0.0308 0.1485 0.238

2023-09-29 18:10:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 18:10:28 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 18:10:28 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 18:10:42 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 18:11:18 [DEBUG] XGB iter   0: tr-p-rmse: 0.295664	tr-a-peak@32: 0.919223	tr-rmse: 0.568510	tr-rmse: 0.568510
2023-09-29 18:11:19 [DEBUG] XGB iter  25: tr-p-rmse: 0.020179	tr-a-peak@32: 1.000000	tr-rmse: 0.623740	tr-rmse: 0.623740
2023-09-29 18:11:19 [DEBUG] XGB iter  50: tr-p-rmse: 0.020179	tr-a-peak@32: 1.000000	tr-rmse: 0.623740	tr-rmse: 0.623740
2023-09-29 18:11:19 [DEBUG] XGB stopped. Best iteration: [8] tr-p-rmse:0.02018	tr-a-peak@32:1.00000	tr-rmse:0.62374	tr-rmse:0.62374 
2023-09-29 18:11:19 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 18:11:19 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1018.3257 |       4.1188 |                4.1188 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.11882

2023-09-29 18:11:19 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 18:11:19 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1018.3257 |       4.1188 |                4.1188 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.11882

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

2023-09-29 18:11:25 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 18:11:25 [INFO] LocalBuilder: max_workers = 56
2023-09-29 18:11:26 [INFO] LocalRunner: max_workers = 1
2023-09-29 18:16:01 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 18:16:01 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 18:16:01 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 18:16:01 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 18:19:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1278 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x995d648)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xfd2a3b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x7d17ba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa48c5f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x95be7f8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x12cc12c8)]: 671 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xa8041b8)]: 0 failure(s)
2023-09-29 18:20:15 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x995d648)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xfd2a3b8)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x7d17ba8)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xa48c5f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x95be7f8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x12cc12c8)]: 671 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xa8041b8)]: 0 failure(s)
2023-09-29 18:20:16 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 99 candidate(s)
2023-09-29 18:34:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9999 0.9914 0.9872 0.9648 0.9412 0.9307 0.9574 0.8741 0.8667 0.6275 0.8948 0.8808 0.9142 0.8971 0.9126 0.8063

[17 : 32]:	0.6245 0.5285 0.7917 0.4139 0.4379 0.8473 0.5379 0.5956 0.7754 0.5783 0.8477 0.8917 0.5673 0.5839 0.7655 0.796

[33 : 48]:	0.2187 0.4102 0.6188 0.4166 0.1195 0.2022 0.5397 0.2994 0.2325 0.0464 0.1111 0.3121 0.4636 0.4432 0.3851 0.305

[49 : 64]:	0.1887 0.124 0.4125 0.1229 0.3941 0.0532 0.8336 0.1836 0.7516 0.0551 0.1623 0.271 0.4557 0.1287 0.5375 0.4334

2023-09-29 18:34:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 18:34:36 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 18:34:37 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 18:34:51 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 18:35:29 [DEBUG] XGB iter   0: tr-p-rmse: 0.278147	tr-a-peak@32: 0.930099	tr-rmse: 0.597477	tr-rmse: 0.597477
2023-09-29 18:35:29 [DEBUG] XGB iter  25: tr-p-rmse: 0.019489	tr-a-peak@32: 0.997533	tr-rmse: 0.651864	tr-rmse: 0.651864
2023-09-29 18:35:29 [DEBUG] XGB iter  50: tr-p-rmse: 0.019489	tr-a-peak@32: 0.997533	tr-rmse: 0.651864	tr-rmse: 0.651864
2023-09-29 18:35:29 [DEBUG] XGB stopped. Best iteration: [12] tr-p-rmse:0.01949	tr-a-peak@32:0.99753	tr-rmse:0.65186	tr-rmse:0.65186 
2023-09-29 18:35:29 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 18:35:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       966.6060 |       4.3392 |                4.3392 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.33921

2023-09-29 18:35:29 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 18:35:29 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       966.6060 |       4.3392 |                4.3392 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.33921

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

2023-09-29 18:35:36 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 18:35:36 [INFO] LocalBuilder: max_workers = 56
2023-09-29 18:35:37 [INFO] LocalRunner: max_workers = 1
2023-09-29 18:40:09 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 18:40:09 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 18:40:10 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 18:40:10 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 18:43:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1405 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7d17fc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x12178798)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xa5a1258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4e5d888)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa17cec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5deb868)]: 564 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x9f87938)]: 0 failure(s)
2023-09-29 18:44:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x7d17fc8)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x12178798)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xa5a1258)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4e5d888)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0xa17cec8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x5deb868)]: 564 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x9f87938)]: 0 failure(s)
2023-09-29 18:44:33 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 79 candidate(s)
2023-09-29 19:01:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9993 0.9413 0.9011 0.9251 0.7856 0.8837 0.894 0.8728 0.6452 0.6886 0.7731 0.7422 0.8686 0.7184 0.7812 0.7771

[17 : 32]:	0.3925 0.467 0.6143 0.2351 0.641 0.5633 0.6082 0.3796 0.7305 0.4801 0.6945 0.5871 0.3283 0.5905 0.7169 0.6183

[33 : 48]:	0.1746 0.2407 0.2644 0.1784 0.2708 0.0935 0.51 0.0293 0.1402 0.1019 0.1602 0.2997 0.4044 0.3353 0.0718 0.0262

[49 : 64]:	0.0799 0.3686 0.6706 0.0876 0.192 0.3673 0.5607 0.1471 0.1801 0.2197 0.3031 0.2232 0.5485 0.1869 0.4423 0.5471

2023-09-29 19:01:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 19:01:05 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 19:01:06 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 19:01:24 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 19:01:59 [DEBUG] XGB iter   0: tr-p-rmse: 0.332602	tr-a-peak@32: 0.848968	tr-rmse: 0.559930	tr-rmse: 0.559930
2023-09-29 19:01:59 [DEBUG] XGB iter  25: tr-p-rmse: 0.029415	tr-a-peak@32: 1.000000	tr-rmse: 0.616531	tr-rmse: 0.616531
2023-09-29 19:02:00 [DEBUG] XGB iter  50: tr-p-rmse: 0.029415	tr-a-peak@32: 1.000000	tr-rmse: 0.616531	tr-rmse: 0.616531
2023-09-29 19:02:00 [DEBUG] XGB stopped. Best iteration: [11] tr-p-rmse:0.02942	tr-a-peak@32:1.00000	tr-rmse:0.61653	tr-rmse:0.61653 
2023-09-29 19:02:00 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 19:02:00 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1021.9337 |       4.1043 |                4.1043 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.10428

2023-09-29 19:02:00 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 19:02:00 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1021.9337 |       4.1043 |                4.1043 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.10428

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

2023-09-29 19:02:06 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 19:02:06 [INFO] LocalBuilder: max_workers = 56
2023-09-29 19:02:07 [INFO] LocalRunner: max_workers = 1
2023-09-29 19:06:36 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 19:06:36 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 19:06:37 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 19:06:37 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 19:10:11 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1469 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xe3ea478)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xadf6588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x16e3b518)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xe5fe6f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x4f31288)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xa4bc618)]: 513 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x967c028)]: 0 failure(s)
2023-09-29 19:10:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xe3ea478)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xadf6588)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x16e3b518)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0xe5fe6f8)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x4f31288)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0xa4bc618)]: 513 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x967c028)]: 0 failure(s)
2023-09-29 19:10:59 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 66 candidate(s)
2023-09-29 19:22:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9999 0.9936 0.9279 0.9884 0.8688 0.8836 0.9132 0.8581 0.781 0.6365 0.7856 0.8235 0.7968 0.7843 0.8571 0.7631

[17 : 32]:	0.7431 0.6458 0.7091 0.3813 0.5605 0.4309 0.7651 0.5257 0.5974 0.5642 0.67 0.5107 0.5569 0.4144 0.4969 0.6259

[33 : 48]:	0.4777 0.3402 0.4383 0.2673 0.6024 0.137 0.595 0.1118 0.1286 0.0511 0.232 0.3466 0.2501 0.5072 0.6036 0.2245

[49 : 64]:	0.2532 0.2978 0.5637 0.4081 0.2471 0.2639 0.449 0.334 0.5085 0.4846 0.2753 0.0396 0.089 0.4625 0.0369 0.0923

2023-09-29 19:22:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 19:22:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 19:22:42 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 19:23:02 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 19:23:43 [DEBUG] XGB iter   0: tr-p-rmse: 0.336969	tr-a-peak@32: 0.865819	tr-rmse: 0.568628	tr-rmse: 0.568628
2023-09-29 19:23:43 [DEBUG] XGB iter  25: tr-p-rmse: 0.036637	tr-a-peak@32: 1.000000	tr-rmse: 0.625930	tr-rmse: 0.625930
2023-09-29 19:23:43 [DEBUG] XGB iter  50: tr-p-rmse: 0.036637	tr-a-peak@32: 1.000000	tr-rmse: 0.625930	tr-rmse: 0.625930
2023-09-29 19:23:43 [DEBUG] XGB stopped. Best iteration: [15] tr-p-rmse:0.03664	tr-a-peak@32:1.00000	tr-rmse:0.62593	tr-rmse:0.62593 
2023-09-29 19:23:43 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 19:23:43 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1000.6888 |       4.1914 |                4.1914 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.19142

2023-09-29 19:23:43 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 19:23:43 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |      1000.6888 |       4.1914 |                4.1914 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.19142

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

2023-09-29 19:23:49 [INFO] Logging directory: ./tune_tmp/logs
2023-09-29 19:23:49 [INFO] LocalBuilder: max_workers = 56
2023-09-29 19:23:50 [INFO] LocalRunner: max_workers = 1
2023-09-29 19:28:20 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-09-29 19:28:20 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-09-29 19:28:21 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-09-29 19:28:21 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:496] Generating candidates......
2023-09-29 19:32:04 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:500] Picked top 1533 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xe48e078)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xab4a408)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb18ef58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x8695228)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x125dc778)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9adfb38)]: 465 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xe75c718)]: 0 failure(s)
2023-09-29 19:32:47 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:381] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0xe48e078)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0xab4a408)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0xb18ef58)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x8695228)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x125dc778)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x9adfb38)]: 465 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0xe75c718)]: 0 failure(s)
2023-09-29 19:32:48 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:512] Sample 50 candidate(s)
2023-09-29 19:43:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:630] Scores of the best 64 schedules:
[1 : 16]:	0.9997 0.9809 0.9455 0.9739 0.9633 0.721 0.8419 0.9611 0.6338 0.917 0.9574 0.5555 0.5292 0.8333 0.8065 0.9037

[17 : 32]:	0.7717 0.1113 0.227 0.6815 0.8471 0.7473 0.8907 0.5066 0.5163 0.2609 0.4231 0.8058 0.6109 0.7029 0.7896 0.8445

[33 : 48]:	0.4437 0.1235 0.4687 0.0135 0.0422 0.0459 0.0197 0.4899 0.6477 0.4804 0.5382 0.5983 0.2048 0.1864 0.6644 0.3495

[49 : 64]:	0.4755 0.3586 0.0693 0.1733 0.1184 0.3336 0.3529 0.1908 0.6176 0.3167 0.479 0.1485 0.6082 0.09 0.1762 0.4453

2023-09-29 19:43:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:517] Got 64 candidate(s) with evolutionary search
2023-09-29 19:43:41 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:520] Sendding 64 candidates(s) for measurement
2023-09-29 19:43:42 [INFO] [task_scheduler.cc:250] Sending 64 sample(s) to builder
2023-09-29 19:44:02 [INFO] [task_scheduler.cc:252] Sending 64 sample(s) to runner
2023-09-29 19:44:38 [DEBUG] XGB iter   0: tr-p-rmse: 0.325146	tr-a-peak@32: 0.941160	tr-rmse: 0.620508	tr-rmse: 0.620508
2023-09-29 19:44:38 [DEBUG] XGB iter  25: tr-p-rmse: 0.032428	tr-a-peak@32: 0.999654	tr-rmse: 0.673343	tr-rmse: 0.673343
2023-09-29 19:44:38 [DEBUG] XGB iter  50: tr-p-rmse: 0.032428	tr-a-peak@32: 0.999654	tr-rmse: 0.673343	tr-rmse: 0.673343
2023-09-29 19:44:38 [DEBUG] XGB stopped. Best iteration: [10] tr-p-rmse:0.03243	tr-a-peak@32:0.99965	tr-rmse:0.67334	tr-rmse:0.67334 
2023-09-29 19:44:38 [INFO] [task_scheduler.cc:294] [Updated] Task #0: "main"
2023-09-29 19:44:38 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       947.3032 |       4.4276 |                4.4276 |     64 |      
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.42763

2023-09-29 19:44:38 [INFO] [task_scheduler.cc:317] Task #0 has finished. Remaining task(s): 0
2023-09-29 19:44:38 [INFO] [task_scheduler.cc:377] 
 ID | Name |    FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
------------------------------------------------------------------------------------------------------
  0 | main | 4194304 |      1 |       947.3032 |       4.4276 |                4.4276 |     64 |    Y 
------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 4.42763

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

