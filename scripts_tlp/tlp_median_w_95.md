2023-10-23 20:58:27 [INFO] Logging directory: ./tune_tmp/logs
2023-10-23 20:58:34 [INFO] LocalBuilder: max_workers = 56
2023-10-23 20:58:38 [INFO] LocalRunner: max_workers = 1
2023-10-23 20:58:47 [INFO] [task_scheduler.cc:210] $$$$$$ This is a test for C++ build $$$$$
2023-10-23 20:58:47 [INFO] [task_scheduler.cc:214] Initializing Task #0: "main"
2023-10-23 20:58:50 [INFO] [task_scheduler.cc:236] TaskScheduler picks Task #0: "main"
2023-10-23 20:58:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:498] Generating candidates......
2023-10-23 20:58:50 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:502] Picked top 0 candidate(s) from database
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x456af08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x4614d08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4504c28)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4513178)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x162f2d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x46117e8)]: 1812 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x44f8db8)]: 0 failure(s)
2023-10-23 21:03:35 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:383] Sample-Init-Population summary:
Postproc #0 [meta_schedule.DisallowDynamicLoop(0x456af08)]: 0 failure(s)
Postproc #1 [meta_schedule.RewriteCooperativeFetch(0x4614d08)]: 0 failure(s)
Postproc #2 [meta_schedule.RewriteUnboundBlock(0x4504c28)]: 0 failure(s)
Postproc #3 [meta_schedule.RewriteParallelVectorizeUnroll(0x4513178)]: 0 failure(s)
Postproc #4 [meta_schedule.RewriteReductionBlock(0x162f2d8)]: 0 failure(s)
Postproc #5 [meta_schedule.VerifyGPUCode(0x46117e8)]: 1812 failure(s)
Postproc #6 [meta_schedule.RewriteTensorize(0x44f8db8)]: 0 failure(s)
2023-10-23 21:03:38 [INFO] [tvm.meta_schedule.search_strategy.gflownet_search:514] Sample 236 candidate(s)
