from src.gfn.mlc_dataset import gflownet_data_save

if __name__ == "__main__":
    # generate dataset for GFlowNet
    # gflownet_data_save("/root/share/dataset/measure_candidate_v2/","/root/share/dataset/gflownet_dataset/")
    gflownet_data_save("/root/share/dataset/debug_measure_candidate",
                       "/root/share/dataset/gflownet_dataset0/")
