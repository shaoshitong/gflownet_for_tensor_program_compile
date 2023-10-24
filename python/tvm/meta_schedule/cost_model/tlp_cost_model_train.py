# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# type: ignore[import]

from torch.utils.data import DataLoader, Dataset
import logging
import glob
import math
import os
import random
import tempfile
from collections import OrderedDict
from itertools import chain as itertools_chain
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
from torch import nn
import time
import pickle
from tqdm import tqdm
import argparse
# import multiprocessing
import json

from torch.nn.utils.rnn import pad_sequence
# from ..utils import derived_object, shash2hex
# from ..logging import get_logger


class FeatureGroup:
    """Feature group

    Parameters
    ----------
    group_hash : str
        The hash of the group
    features : List[np.ndarray]
        The features
    costs : List[float]
        The costs
    min_cost : float
        The minimum cost
    """

    group_hash: str
    features: List[np.ndarray]
    costs: np.ndarray
    min_cost: float

    def __init__(
        self,
        group_hash: str,
        features: List[np.ndarray],
        costs: np.ndarray,
        min_costs=None
    ) -> None:
        self.group_hash = group_hash
        self.features = features
        self.costs = costs
        self.min_cost = np.min(costs) if min_costs == None else min_costs

    def append(  # pylint: disable=missing-function-docstring
        self,
        features: List[np.ndarray],
        costs: np.ndarray,
    ) -> None:
        self.features.extend(features)
        self.costs = np.append(self.costs, costs)
        self.min_cost = np.min(self.costs)

    def to_json(self):
        return {"features": [feature.tolist() for feature in self.features], "mean_costs": self.costs.tolist(), "min_cost": float(self.min_cost)}

    @classmethod
    def from_json(cls, json_obj):
        key = list(json_obj.keys())[0]
        data = list(json_obj.values())[0]
        return cls(key, data["features"], data["mean_costs"], data["min_cost"])


# logger = get_logger("transformer")  # pylint: disable=invalid-name


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(message)s")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join(".", "train_log")
    log_path = work_dir + '/log.txt'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if os.path.exists(log_path):
        os.remove(log_path)
    fHandler = logging.FileHandler(log_path, mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger

# pylint: disable=too-many-instance-attributes


class SegmentDataLoader:
    """Dataloader for Segment Sum MLP model.

    Parameters
    ----------
    features : List[np.ndarray]
        The features
    results : np.ndarray
        The measured results, can be None.
    batch_size : int
        The batch size
    shuffle : bool
        Whether to shuffle the dataset or not
    """

    def __init__(
        self,
        features,
        results=None,
        batch_size=128,
        shuffle=True,
    ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(features)

        self.iter_order = self.pointer = None

        self.data_steps = pad_sequence(
            [torch.tensor(f) for f in features], batch_first=True)
        # NOTE: fix bug for results is None in tlp predict
        self.labels = None
        if not results is None:
            self.labels = torch.FloatTensor(results)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.data_size)
        else:
            self.iter_order = torch.arange(self.data_size)
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer >= self.data_size:
            raise StopIteration
        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):

        batch_datas_steps = self.data_steps[indices]
        # batch_datas_steps = nn.utils.rnn.pad_sequence(
        #     batch_datas_steps, batch_first=True)

        if str(torch.__version__)[0] == '0':
            batch_datas_steps = nn.utils.rnn.pad_sequence(
                batch_datas_steps, batch_first=True)
        else:
            batch_datas_steps = nn.utils.rnn.pad_sequence(
                [s[0] for s in torch.split(batch_datas_steps, 1, 0)], batch_first=True)

        # NOTE: fix bug for self.labels is None in tlp predict
        batch_labels = 0
        if not self.labels is None:
            batch_labels = self.labels[indices]

        return (batch_datas_steps, batch_labels)


class SegmentDataset(Dataset):
    def __init__(self,
                 features,
                 results=None):
        self.data_size = len(features)
        self.data_steps = pad_sequence(
            [torch.tensor(f) for f in features], batch_first=True)
        if results is None:
            self.labels = torch.Tensor([0 for _ in range(self.data_size)])
        else:
            self.labels = torch.Tensor(results)

    def __len__(self):
        return self.data_size

    def __getitem__(self, indices):
        batch_datas_steps = self.data_steps[indices]
        if str(torch.__version__)[0] == '0':
            batch_datas_steps = nn.utils.rnn.pad_sequence(
                batch_datas_steps[None, ...], batch_first=True)[0]
        else:
            batch_datas_steps = nn.utils.rnn.pad_sequence(
                (batch_datas_steps,), batch_first=True)[0]
        batch_labels = self.labels[indices]

        return (batch_datas_steps, batch_labels)


def SegmentDataloder_new(features, results=None, batch_size=128, shuffle=True, num_worker=4):
    dataset = SegmentDataset(features, results)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_worker)
    return dataloader


class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                                          None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


class TransformerModule(torch.nn.Module):
    fea_size: int
    step_size: int

    def __init__(self, fea_size=172, step_size=25) -> None:
        super().__init__()
        self.fea_size = fea_size
        self.step_size = step_size

        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_1, dim_feedforward=256, nhead=args.attention_head
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):
        # batch_datas_steps = batch_datas_steps[:self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.transformer(encoder_output)

        output = self.decoder(output).sum(0)

        return output.squeeze()


def validate(model, valid_loader, loss_func, device, name=""):
    # import wandb
    # wandb_project = "Validate TLP with hardware"
    # use_wandb = len(wandb_project) > 0
    # if use_wandb:
    #     wandb.init(project=wandb_project)
    #     wandb.config.update({
    #         "architecture": "TLP Cost Model",
    #         "dataset": "Dataset of Extract Features",

    #     })
    model.eval()
    valid_losses = []

    for batch_data, batch_label in valid_loader:
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        preds = model(batch_data)
        labels = batch_label
        valid_loss = loss_func(preds, labels)
        # cc = 0
        # for pred, label in zip(preds, labels):
        #     cc += 1
        #     if cc > 100:
        #         break
        #     # wandb.log({f"{name} prediction": pred, f"{name} label": label})
        valid_losses.append(valid_loss.item())

    return np.sum(valid_losses)/len(valid_loader)


def from_json(path: str) -> List[FeatureGroup]:
    assert os.path.exists(path)
    datasets = []
    json_files = glob.glob(os.path.join(path, "*.json"))
    for json_file in json_files:
        obj = None
        with open(json_file, 'r') as f:
            obj = json.load(f)
        feature_group = FeatureGroup.from_json(obj)
        assert len(feature_group.features) == len(feature_group.costs)
        datasets.append(feature_group)
    return datasets


def load_data(logger, datasets_all):
    indices = np.array(range(len(datasets_all)))
    train_len = int(len(indices) * 0.9)

    data_shuffle = np.random.permutation(indices)
    train_idx, val_idx = data_shuffle[:train_len], data_shuffle[train_len:]

    train_data = []
    val_data = []

    for idx in indices:
        if np.isin(idx, train_idx):
            train_data.append(datasets_all[idx])
        else:
            val_data.append(datasets_all[idx])

    train_feature = list(
        itertools_chain.from_iterable([g.features for g in train_data])
    )
    val_feature = list(
        itertools_chain.from_iterable([g.features for g in val_data])
    )
    # # compute scores
    # train_label = np.concatenate(
    #     [g.min_cost / np.array(g.costs) for g in train_data])
    # val_label = np.concatenate(
    #     [g.min_cost / np.array(g.costs) for g in val_data])
    # compute scores
    train_label = np.concatenate(
        [np.array(g.costs) for g in train_data])
    val_label = np.concatenate(
        [np.array(g.costs) for g in val_data])

    logger.info("train_data length is %d", len(train_feature))
    logger.info("val_data length is %d", len(val_feature))

    n_gpu = torch.cuda.device_count()
    train_dataloader = SegmentDataLoader(
        train_feature, train_label, args.train_size_per_gpu * n_gpu, True)
    val_dataloader = SegmentDataLoader(
        val_feature, val_label, args.train_size_per_gpu * n_gpu, False
    )

    return train_dataloader, val_dataloader


def eval(logger, datasets_all):

    val_feature = list(
        itertools_chain.from_iterable([g.features for g in datasets_all])
    )

    val_label = np.concatenate(
        [np.array(g.costs) for g in datasets_all])

    logger.info("val_data length is %d", len(val_feature))

    n_gpu = torch.cuda.device_count()

    valid_loader = SegmentDataLoader(
        val_feature, val_label, args.train_size_per_gpu * n_gpu, False
    )

    device = "cuda"
    loss_func = LambdaRankLoss(device)
    tlp_median_w_95_path = "/root/kongdehao/model/0test_tlp/tlp_median_w_95.pth"
    tlp_median_w_95 = torch.load(tlp_median_w_95_path, map_location=device)
    tlp_median_w_95.to(device)

    tlp_v2_mean_72_path = "/root/kongdehao/model/0test_tlp/tlp_v2_mean_72.pth"
    tlp_v2_mean_72 = torch.load(tlp_v2_mean_72_path, map_location=device)
    tlp_v2_mean_72.to(device)

    tlp_v2_median_w_58_path = "/root/kongdehao/model/0test_tlp/tlp_v2_median_w_58.pth"
    tlp_v2_median_w_58 = torch.load(
        tlp_v2_median_w_58_path, map_location=device)
    tlp_v2_median_w_58.to(device)

    tlp_mean_88_path = "/root/kongdehao/model/0test_tlp/tlp_mean_88.pth"
    tlp_mean_88 = torch.load(tlp_mean_88_path, map_location=device)
    tlp_mean_88.to(device)

    tlp_median_home_14_path = "/root/kongdehao/model/tlp/median/tlp_median_home_14.pth"

    tlp_median_home_14 = torch.load(
        tlp_median_home_14_path, map_location=device)
    tlp_median_home_14.to(device)

    tlp_median_14_path = "/root/kongdehao/model/tlp/median/tlp_median_14.pth"
    tlp_median_14 = torch.load(tlp_median_14_path, map_location=device)
    tlp_median_14.to(device)

    tlp_median_home0_13_path = "/root/kongdehao/model/tlp/median/tlp_median_home0_13.pth"
    tlp_median_home0_13 = torch.load(
        tlp_median_home0_13_path, map_location=device)
    tlp_median_home0_13.to(device)

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
    loss = {}
    tlp_median_w_95_loss = validate(
        tlp_median_w_95, valid_loader, loss_func, device, "tlp_median_w_95")
    tlp_v2_mean_72_loss = validate(
        tlp_v2_mean_72, valid_loader, loss_func, device, "tlp_v2_mean_72")
    tlp_v2_median_w_58_loss = validate(
        tlp_v2_median_w_58, valid_loader, loss_func, device, "tlp_v2_median_w_58")
    tlp_mean_88_loss = validate(
        tlp_mean_88, valid_loader, loss_func, device, "tlp_mean_88")

    tlp_median_home_14_loss = validate(
        tlp_median_home_14, valid_loader, loss_func, device, "tlp_median_home_14")
    tlp_median_14_loss = validate(
        tlp_median_14, valid_loader, loss_func, device, "tlp_median_14")
    tlp_median_home0_13_loss = validate(
        tlp_median_home0_13, valid_loader, loss_func, device, "tlp_median_home0_13")
    tlp_old_14_loss = validate(
        tlp_old_14, valid_loader, loss_func, device, "tlp_old_14")
    loss["tlp_median_w_95_loss"] = tlp_median_w_95_loss
    loss["tlp_v2_mean_72_loss"] = tlp_v2_mean_72_loss
    loss["tlp_v2_median_w_58_loss"] = tlp_v2_median_w_58_loss
    loss["tlp_mean_88_loss"] = tlp_mean_88_loss
    loss["tlp_median_home_14_loss"] = tlp_median_home_14_loss
    loss["tlp_median_14_loss"] = tlp_median_14_loss
    loss["tlp_median_home0_13_loss"] = tlp_median_home0_13_loss
    loss["tlp_old_14_loss"] = tlp_old_14_loss
    # loss = dict(sorted(loss))

    save_path = "/root/kongdehao/model/0test_tlp/0records_dataset_loss.md"
    with open(save_path, "a") as f:
        f.write("extract_features_median")
        for key, val in loss.items():
            f.write(f"{key} = {val}\n")


def train(train_loader, val_dataloader, device, logger):
    import wandb

    net = TransformerModule().to(device)
    # NOTE: specify device id order
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    loss_func = LambdaRankLoss(device)
    # NOTE:
    # loss_func = nn.MSELoss()

    n_epoch = args.n_epoch
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_epoch // 3, gamma=1)

    wandb_project = "train TLP Cost Model with Rank loss"
    use_wandb = len(wandb_project) > 0
    if use_wandb:
        wandb.init(project=wandb_project)
        wandb.config.update({
            "learning_rate": args.lr,
            "architecture": "TLP Cost Model",
            "dataset": "Dataset of Extract Features",
            "epochs": n_epoch,
        })

    logger.info("start train...")
    best_net = net
    min_loss = 1e5
    best_it = 0
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss = 0
        for batch, (batch_data, batch_label) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            # logger.info(f"shape: {batch_data.shape} dim0: {type(batch_data[0])}, type: {batch_data.dtype}")
            loss = loss_func(net(batch_data), batch_label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()

        train_time = time.time() - tic
        avg_train_loss = train_loss/len(train_loader)
        valid_loss = validate(net, val_dataloader, loss_func, device)
        loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (
            avg_train_loss, valid_loss)
        logger.info(
            f"Epoch: {epoch}\tBatch: {batch}\t{loss_msg}\tTrian Speed: {len(train_loader)/train_time}")
        wandb.log({"Epoch": epoch, "Batch": batch, "Train Speed": len(
            train_loader)/train_time, "Train Loss": avg_train_loss, "Valid Loss": valid_loss})

        if valid_loss < min_loss:
            best_net = net
            best_it = epoch
            min_loss = valid_loss

    # checkpoint = {"tlp" : best_net.state_dict()}
    torch.save(best_net, "%s/tlp_median_w_%d.pth" %
               (args.save_model_path, best_it))
    # with open(save_model_path, 'wb') as f:
    #     pickle.dump(best_net.cpu(), f)
    # net.to(device)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: We can convert to cuda:0, but retrain pkl model which cuda is cuda:7
    # model & data are not on same device
    parser.add_argument("--cuda", type=str, default='cuda:0')
    # NOTE: Use you defined dataset path
    parser.add_argument("--dataset_path", type=str,
                        default='/root/share/dataset/0test_extract_features/extract_features_median')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default='default')
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--data_cnt", type=int, default=-1)  # data_cnt * 1000

    parser.add_argument("--train_size_per_gpu", type=int, default=256)
    parser.add_argument("--val_size_per_gpu", type=int, default=256)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--target", type=str, default="nvidia/nvidia-a100")
    parser.add_argument("--save_model_path", type=str,
                        default="/root/kongdehao/model/0test_tlp")

    args = parser.parse_args()

    logger = getLogger()
    logger.info(args)

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    datasets = from_json(path=args.dataset_path)
    # train_loader, val_loader = load_data(logger, datasets)
    # eval(val_loader)
    eval(logger, datasets)
    # train(train_loader, val_loader, device=args.cuda, logger=logger)
