from dataset_embedding import load_all_files,check_decision_same,GflowNetEmbedding

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


    databases = load_all_files("/home/imagenet/share/dataset/measure_candidate_v2")
    
    import copy,tvm
    from tvm.tir.schedule import InstructionKind
    from dataset_embedding import deep_copy_map
    max_value = 0
    for database in databases:
        records = database.get_all_tuning_records()
        for record in records:
            sub_sch = record.as_measure_candidate().sch
            sub_trace = sub_sch.trace
            sub_insts = sub_trace.insts
            sub_decisions = sub_trace.decisions
            AA = (list(deep_copy_map(sub_trace.decisions).values()))
            from tvm.tir.schedule import InstructionKind
            gm = GflowNetEmbedding()
            embedding_results,embedding_conditions,count_ptr_list = gm(sub_insts,sub_decisions,True)
            # Now, disturb the decision's value
            
            for er in embedding_results:
                print(er.shape,end=" ")
            print("")
            new_sub_decisions = deep_copy_map(sub_decisions)
            for key,value in new_sub_decisions.items():
                if key.kind == InstructionKind.get("SampleCategorical"):
                    new_sub_decisions[key] = tvm.tir.const(0, dtype='int32')
                    
            max_value = max(max_value,len(embedding_results))
            new_sub_insts, new_sub_decisions = gm(sub_insts,sub_decisions,False,embedding_results=embedding_results,embedding_conditions=embedding_conditions,count_Ptr_results=count_ptr_list)
            for new_sub_inst,new_sub_decision in zip(new_sub_insts,new_sub_decisions):
                sub_trace.with_decision(new_sub_inst,new_sub_decision,True)
            BB = list(sub_trace.decisions.values())
            print(check_decision_same(AA,BB),max_value)
            assert check_decision_same(AA,BB), "Not same"