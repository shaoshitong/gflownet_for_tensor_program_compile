from src.gfn.mlc_dataset import *
from src.gfn.mlc_dataset.dataset_embedding.load_dataset_meta_schedule import *

if __name__ == "__main__":
    # compute_rankloss()
    measure_tlp("/root/share/dataset/decode_info",
                "/root/kongdehao/model/0test_tlp")
    # measure_tlp("/root/share/dataset/measure_candidate_v2/", "/root/kongdehao/model/0test_tlp")

    # data_path = "/root/share/dataset/measure_candidate_v2/"
    # data_path = "/root/share/dataset/debug_measure_candidate"
    # databases = load_all_files(data_path)
    # for database in databases:
    #     # database made up of records, including candidates info
    #     records = database.get_all_tuning_records()
    #     print(f"len of records = {len(records)}")
