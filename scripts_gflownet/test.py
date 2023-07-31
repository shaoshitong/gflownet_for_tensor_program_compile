from dataset_embedding import get_useful_dicts,get_useful_keys,load_all_files,EmbeddingSamplePerfectTile,EmbeddingAnnotation

if __name__ == "__main__":
    import numpy as np

    def expand_to_binary(array):
        # 找到数组中的最大值
        max_val = array.max()

        # 计算最大值的二进制长度
        max_len = int(np.ceil(np.log2(max_val + 1)))

        # 创建一个新的全零数组，形状为原数组形状加上最大二进制长度
        binary_array = np.zeros(array.shape + (max_len,), dtype=np.uint8)

        # 使用二进制表示填充新数组
        for idx in np.ndindex(array.shape):
            binary_array[idx] = list(f'{array[idx]:0{max_len}b}')[::-1]

        return binary_array


    databases = load_all_files("/home/tvm/scripts_gflownet/dataset/measure_candidate/bert_base-None-1,64")
    for database in databases:
        records = database.get_all_tuning_records()
        for record in records:
            sub_sch = record.as_measure_candidate().sch
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions
            from tvm.tir.schedule import InstructionKind
            for sub_inst in sub_insts:
                if sub_inst.kind == InstructionKind.get("SampleCategorical"):
                    probs = [i.value for i in sub_inst.attrs[1]]
                    values = [i.value for i in sub_inst.attrs[0]]
                    print(probs,values,len(values))
            
            EmbeddingAnnotation.embedding_annotation(sub_insts,sub_decisions)
            
            
            
            # embedding_results,embedding_conditions = EmbeddingSamplePerfectTile.embedding_sample_perfectile(sub_insts,sub_decisions)
            # # Now, disturb the decision's value
            # import copy,tvm
            # from tvm.tir.schedule import InstructionKind
            # from dataset_embedding import deep_copy_map
            
            # new_sub_decisions = deep_copy_map(sub_decisions)
            # for key,value in new_sub_decisions.items():
            #     if key.kind == InstructionKind.get("SamplePerfectTile"):
            #         new_sub_decisions[key] = [tvm.tir.const(1, dtype='int32') for v in value]
            
            # new_sub_insts, new_sub_decisions = EmbeddingSamplePerfectTile.unembedding_sample_perfectile(sub_insts,new_sub_decisions,embedding_results,embedding_conditions)
            # print(new_sub_decisions)
            # for new_sub_inst,new_sub_decision in zip(new_sub_insts,new_sub_decisions):
            #     sub_trace.with_decision(new_sub_inst,new_sub_decision,True)
            # print(sub_trace.decisions)
            
            # if len(new_sub_insts)!=0:
            #     print("="*120)