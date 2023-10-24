if __name__ == "__main__":
    import pickle
    import torch
    from tvm.meta_schedule.cost_model.tlp_cost_model_train import *
    device = "cuda"
    tlp_old_14_path = "/root/kongdehao/model/median_tlp/save_model_v1/tlp_model_14.pkl"
    with open(tlp_old_14_path, 'rb') as f:
        tlp_old_14 = pickle.load(f)
    tlp_old_14.to(device)

    # Modify the device_ids
    tlp_old_14.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # Modify the src_device_obj
    tlp_old_14.src_device_obj = torch.device('cuda:0')
    # Modify the output_device
    tlp_old_14.output_device = torch.device('cuda:0')
    torch.save(
        tlp_old_14, "/root/kongdehao/model/median_tlp/save_model_v1/tlp_old_14.pth")
