from dataset_embedding import load_all_files, check_decision_same, GflowNetEmbedding

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

    # NOTE: change to you data path
    # 每个candidate对应一个workload，一个网络结构 对应多个workload
    # 不同的candidates 差异存在于primitive和decision
    databases = load_all_files("/root/share/dataset/decode_info")

    import copy
    import tvm
    from tvm.tir.schedule import InstructionKind
    from dataset_embedding import deep_copy_map
    max_value = 0
    flag = True
    counter = 0
    for database in databases:
        if not flag:
            break
        records = database.get_all_tuning_records()
        for record in records:
            sub_sch = record.as_measure_candidate().sch
            # trace include instructions & decisions
            sub_trace = sub_sch.trace
            # instructions include deterministic and stochastic
            sub_insts = sub_trace.insts
            # decision only include stochastic instructions
            sub_decisions = sub_trace.decisions
            old_decisions = dict(sub_decisions)
            # print(f"old insts & decisions = {sub_decisions}")
            AA = (list(deep_copy_map(sub_trace.decisions).values()))
            # print(f"old decision list = {AA}")
            from tvm.tir.schedule import InstructionKind

            gm = GflowNetEmbedding()
            # True: encode()
            embedding_results, embedding_conditions, count_ptr_list, new_decisions = gm(
                sub_insts, sub_decisions, True)
            # Now, disturb the decision's value
            # new_insts = [eval(k) for k in new_decisions.keys()]
            # new_decis = []
            # for val in new_decisions.values():
            #     if isinstance(val, list):
            #         new_decis.append([tvm.tir.const(i, dtype='int32') for i in val])
            #     else:
            #         new_decis.append(tvm.tir.const(val, dtype='int32'))


            if len(embedding_results) != len(AA):
                print(
                    f"Wrong len of res = {len(embedding_results)}, len of decision = {len(AA)}")
                print(embedding_results)
                print(AA)
                flag = False
                break
            new_sub_decisions = dict(sub_decisions)
            for key, value in new_sub_decisions.items():
                if key.kind == InstructionKind.get("SampleCategorical"):
                    new_sub_decisions[key] = tvm.tir.const(
                        -1, dtype='int32')
                if key.kind == InstructionKind.get("SamplePerfectTile"):
                    new_sub_decisions[key] = [tvm.tir.const(i, dtype='int32')
                                              for i in range(len(value))]

            print(
                f"random decision list = {list(new_sub_decisions.values())}")
            # NOTE: following is training & model
            max_value = max(max_value, len(embedding_results))
            # False: decode(), new_insts & sub_decisions are list
            # new_sub_insts, new_sub_decisions = gm(sub_insts, sub_decisions, False, embedding_results=embedding_results,
            #                                       embedding_conditions=embedding_conditions, count_Ptr_results=count_ptr_list)
            new_sub_decisions = gm(sub_insts, new_sub_decisions, False, embedding_results=embedding_results,
                                   embedding_conditions=embedding_conditions, count_Ptr_results=count_ptr_list)

            print(f"new decision list = {list(new_sub_decisions.values())}")
            # Must use with_decision() to set sub_trace
            # NOTE: new_sub_decisions is same as old sub_decisions, but sub_trace is arbitrary order
            for new_sub_inst, new_sub_decision in new_sub_decisions.items():
                sub_trace = sub_trace.with_decision(
                    new_sub_inst, new_sub_decision, True)

            # print(f"len of old decision = {len(AA)}")
            BB = list(sub_trace.decisions.values())
            # print(f"new sub_trace decision list = {BB}")
            # print(f"len of new decision = {len(BB)}")
            # print(check_decision_same(sub_decisions, sub_trace.decisions), max_value)
            from collections import Counter
            if Counter(sub_decisions) != Counter(new_sub_decisions):
                # if Counter(sub_decisions) != Counter(sub_trace.decisions):
                print(f"Not same decisions")
                # print(f"old insts = {list(sub_decisions.keys())}")
                # print(f"new insts = {list(sub_trace.decisions.keys())}")

            print(f"old decision = {list(sub_decisions.values())}")
            print(f"update decision = {list(sub_trace.decisions.values())}")

            # decisions1 = dict(old_decisions)
            # decisions2 = dict(sub_trace.decisions)
            # for key, v1 in decisions1.items():
            #     v2 = decisions2[key]
            #     if type(v1) != type(v2):
            #         print("types are not same", type(v1), type(v2))
            #         flag = False
            #         break
            #     if isinstance(v1, (int, tvm.tir.expr.IntImm)):
            #         if v1 != v2:
            #             print("ints are not same", v1, v2)
            #             flag = False
            #             break
            #     elif isinstance(v1, (list, tvm.ir.container.Array)):
            #         if v1 != v2:
            #             print("lists are not same", v1, v2)
            #             flag = False
            #             break

            print(f"Finish {counter}")
            counter += 1
