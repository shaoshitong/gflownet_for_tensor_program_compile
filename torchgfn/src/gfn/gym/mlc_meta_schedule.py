from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, cast

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from src.gfn.actions import Actions
from src.gfn.env import DiscreteEnv
from src.gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from src.gfn.states import DiscreteStates, States

import numpy as np
from tvm.runtime import NDArray
from tvm.meta_schedule.feature_extractor import FeatureExtractor, PerStoreFeature
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.tune_context import TuneContext
from typing import Dict, List, NamedTuple, Optional, Tuple
from tvm.meta_schedule.cost_model.tlp_cost_model_train import *


# NOTE: need for double check for this file -- To implement MetaSchedule Env
class EnergyFunction(nn.Module, ABC):
    """Base class for energy functions"""

    @abstractmethod
    def forward(
        self, states: TT["batch_shape", "state_shape", torch.float]
    ) -> TT["batch_shape"]:
        pass


class IsingModel(EnergyFunction):
    """Ising model energy function"""

    def __init__(self, J: TT["state_shape", "state_shape", torch.float]):
        super().__init__()
        self.J = J
        self.linear = nn.Linear(J.shape[0], 1, bias=False)
        self.linear.weight.data = J

    def forward(
        self, states: TT["batch_shape", "state_shape", torch.float]
    ) -> TT["batch_shape"]:
        states = states.float()
        tmp = self.linear(states)
        return -(states * tmp).sum(-1)


class MetaScheduleEnv(DiscreteEnv):
    def __init__(self,
                 cond1_ndim=32,
                 one_hot_len=10,
                 cond1_len=15,
                 cond0_len=15,
                 energy=None,
                 alpha: float = 1.0,
                 device_str="cpu",
                 preprocessor_name="Identity"):

        # action's [0,one_hot_len*cond0_len-1]==[0, 15*10-1] represents the choice \
        # of cuda_bind and annotation, while action's [one_hot_len*one_hot \
        # _seq_len,one_hot_len*cond0_len+2*cond1_ndim*binary_seq_l \
        # en+1]==[15*10, 15*10+2*15*96-1] represents the choice of sample_perfectile, while action's  \
        # [one_hot_len*cond0_len+2*cond1_ndim*cond1_len+1] rep \
        # resents the choice of termination condition.

        # The dimension of the state is $\mathbb{R}^{cond0_len+binary \
        # _seq_len*cond1_ndim}$==15+15*96, where the first cond0_len values ar \
        # e in the range of [0,one_hot_len-1]==[0, 9] and the inverse cond1_len \
        # *cond1_ndim values are in the range of [0,1].

        self.cond1_ndim = cond1_ndim
        self.one_hot_len = one_hot_len
        self.cond1_len = cond1_len
        self.cond0_len = cond0_len
        # ndim = 15+15*32
        self.ndim = self.cond0_len+self.cond1_len*self.cond1_ndim
        # n_action = 15*10 + 15*32*10
        n_actions = self.one_hot_len*self.cond0_len + \
            self.one_hot_len*self.cond1_ndim*self.cond1_len + 1

        s0 = torch.full((self.ndim,), fill_value=-1,
                        dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((self.ndim,), fill_value=-2,
                        dtype=torch.long, device=torch.device(device_str))

        if energy is None:
            energy = IsingModel(
                torch.ones((self.ndim, self.ndim),
                           device=torch.device(device_str))
            )
        self.energy: EnergyFunction = energy
        self.alpha = alpha

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_dim=self.ndim)
        elif preprocessor_name == "Enum":
            preprocessor = EnumPreprocessor(
                get_states_indices=self.get_states_indices,
            )
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_States_class(self):
        env = self

        class DiscreteMSStates(DiscreteStates):
            state_shape = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions
            device = env.device

            # # NOTE: get random state -- val in [-1, 0, .., ndim-1]
            # @classmethod
            # def make_random_states_tensor(
            #     cls, batch_shape: Tuple[int, ...]
            # ) -> TT["batch_shape", "state_shape", torch.float]:
            #     one_hot_seq = torch.randint(-1,
            #                                 env.one_hot_len,  # NOTE: high is ndim, not ndim-1
            #                                 batch_shape + \
            #                                 (env.cond0_len, ),
            #                                 dtype=torch.long,
            #                                 device=env.device)
            #     binary_seq = torch.randint(-1,
            #                                2,
            #                                batch_shape +
            #                                (env.cond1_len*env.cond1_ndim, ),
            #                                dtype=torch.long,
            #                                device=env.device)
            #     # As 0th dim is batch_shape, cat in 1th dim
            #     total_seq = torch.cat([one_hot_seq, binary_seq], 1)

            #     return total_seq

            def make_masks(
                self,
            ) -> Tuple[
                TT["batch_shape", "n_actions", torch.bool],
                TT["batch_shape", "n_actions - 1", torch.bool],
            ]:
                forward_masks = torch.zeros(
                    self.batch_shape + (env.n_actions,),
                    device=env.device,
                    dtype=torch.bool,
                )
                backward_masks = torch.zeros(
                    self.batch_shape + (env.n_actions - 1,),
                    device=env.device,
                    dtype=torch.bool,
                )

                return forward_masks, backward_masks

            def update_masks(self, info) -> None:

                # The following two lines are for typing only.

                self.forward_masks = cast(
                    TT["batch_shape", "n_actions", torch.bool],
                    self.forward_masks,
                )
                self.backward_masks = cast(
                    TT["batch_shape", "n_actions - 1", torch.bool],
                    self.backward_masks,
                )

                '''
                # 1) Both forward and backward masks need to be separated according \
                # to the different parts of the action. This refers to the three di \
                # fferent cases of annotation,sample_perfectile,cuda_bind. In a for \
                # ward mask, state[:cond0_len] is selectable as long as it is \
                # not -1. Similarly, state[cond0_len:] is selectable as long \
                # as it is not -1.
                '''

                act_low0 = env.one_hot_len*env.cond0_len
                act_high0 = act_low0 + env.one_hot_len*env.cond1_ndim*env.cond1_len
                sta_low0 = env.cond0_len
                valid_mask = torch.full_like(
                    self.forward_masks, fill_value=1, device=self.tensor.device).bool()

                if info != None:

                    # shape: (16) (16, 7) (16, 15) (16, 30, 34) (16, 3)
                    databases_path, decode, order, cond, ptr, target = info

                    cond_x0, cond_y0, cond_x1, cond_y1, max_len, emb0_x, emb1_x = torch.split(
                        decode, split_size_or_sections=1, dim=1)

                    ex_cond0, ex_cond1 = torch.split(
                        cond, split_size_or_sections=15, dim=1)  # split along 1dim
                    cond0 = []
                    cond1 = []
                    for i in range(cond_x0.shape[0]):
                        t0, _ = torch.split(
                            ex_cond0[i], [cond_x0[i].item(), 15-cond_x0[i].item()], 0)
                        t1, _ = torch.split(
                            ex_cond1[i], [cond_x1[i].item(), 15-cond_x1[i].item()], 0)

                        t0, _ = torch.split(
                            t0, [cond_y0[i].item(), 34-cond_y0[i].item()], 1)
                        t1, _ = torch.split(
                            t1, [cond_y1[i].item(), 34-cond_y1[i].item()], 1)
                        cond0.append(t0)
                        cond1.append(t1)

                    for id in range(len(cond0)):  # (16)
                        if cond0[id].shape[0] > 0:
                            old_len0 = cond0[id][..., 3].to("cuda")  # (3, 24)
                            n = len(cond0)
                            m = old_len0.shape[0]
                            # NOTE: must satisfy all condition
                            for i in range(m):
                                hi = 10*i + old_len0[i].item()
                                ma = 10 * (i+1)
                                valid_mask[..., id, int(hi):ma] = False
                        # NOTE: init zero for greater than len(factors)
                        import numpy as np

                        # TODO: satisfy cond1
                        if cond1[id].shape[0] > 0:
                            old_len1 = cond1[id][..., 0].to("cuda")  # (3, 1)
                            m = old_len1.shape[0]
                            scale = env.one_hot_len*env.cond1_ndim
                            base = act_low0
                            # NOTE: must satisfy all condition
                            for i in range(m):
                                tt = cond1[id][i].numpy()  # (4, 34)
                                factors = tt[2:]
                                ed = np.argmin(factors != 0)
                                factors = factors[:ed]
                                lenf = len(factors)
                                for j in range(lenf):
                                    lw = base + scale * i + j*env.one_hot_len
                                    # invalid range: [0] U [num+1, ma]
                                    # NOTE: mask res == 0, 0 is valid
                                    valid_mask[..., id, lw] = False
                                    # mask [num+1, ma]
                                    hi = base + scale * i + j * \
                                        env.one_hot_len + \
                                        old_len1[i].item() + 1
                                    ma = base + scale * i + \
                                        (j+1) * env.one_hot_len
                                    valid_mask[..., id, int(
                                        hi):int(ma)] = False

                # NOTE: forward mask is valid position flag (-1) for action
                # [..., None]: add new dim, [1, 2, 3] --> [[1], [2], [3]]
                # expand(): convert one flag state into ndim action -- (15, ) --> (15, 10)
                # NOTE: for traj: (s0, s1, ... sf)
                self.forward_masks[..., :act_low0] = \
                    (self.tensor[..., :sta_low0] == -1)[..., None].expand(*self.tensor.shape[:-1],
                                                                          env.cond0_len, env.one_hot_len).contiguous().view(*self.tensor.shape[:-1], act_low0)

                self.forward_masks[..., :act_low0] = torch.logical_and(
                    valid_mask[..., :act_low0], self.forward_masks[..., :act_low0])

                self.forward_masks[..., act_low0:act_high0] = \
                    (self.tensor[..., sta_low0:] == -1)[..., None].expand(*self.tensor.shape[:-1],
                                                                          env.cond1_ndim*env.cond1_len, env.one_hot_len).contiguous().view(*self.tensor.shape[:-1], act_high0-act_low0)

                self.forward_masks[..., act_low0:act_high0] = torch.logical_and(
                    valid_mask[..., act_low0:act_high0], self.forward_masks[..., act_low0:act_high0])
                # 2) The last action is not a terminal state as long as one of the e \
                # xistent states is -1.
                # torch.all(input): test if all eles in input evaluate to True
                self.forward_masks[..., -
                                   1] = torch.all(self.tensor != -1, dim=-1)

                r'''# 3) The case of backward's mask is more complex, and we need to emp \
                # loy torch.scatter to insert executable markers at the appropriate  \
                # index positions. Specifically, when there is a value other than -1 \
                # in the previous cond0_len values, then there is an action th \
                # at cannot be selected. Assuming that the action corresponds to pos \
                # ition K of cond0_len, then it is computed as [K*one_hot_len \
                # +action_value].
                '''

                # NOTE: backward mask is valid position flag (not -1)
                # src: (16, 15) tar: (16, 150) or traj: (72, 16, 15) (72, 16, 150)
                src0 = torch.full_like(
                    self.tensor[..., :env.cond0_len], fill_value=1, dtype=torch.bool, device=self.tensor.device)
                tar0 = torch.zeros_like(
                    self.backward_masks[..., :act_low0], device=self.tensor.device).bool()
                # convert [0, 5, 2, 7] into 0 + (5 + 10) + (2 + 10 + 10) ...
                # 0 is first 10 pos, 5 is second 10 pos, 2 is third 10 pos
                # (bs, len) + (1, len) == 将arange中一行元素依次累加到bs行，实现了上面的0+(5+10)+(2+10+10)
                index0 = self.tensor[..., :env.cond0_len] + (torch.arange(
                    env.cond0_len, device=self.tensor.device) * env.one_hot_len)[None, ...]
                mask0 = self.tensor[..., :env.cond0_len] >= 0

                # Using advanced indexing instead of the double loop
                # get pos of each nonzero: (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
                valid_indices0 = mask0.nonzero(as_tuple=True)
                # NOTE: only set [0, 5, 2, 7] pos into True, rest is False
                # tar[i,index[i,j]] = src[i,j]
                if len(valid_indices0) == 2:
                    # tuple + same shape tuple = cat(tuple, tuple) --
                    tar0[tuple(valid_indices0[:-1]) +
                         (index0[valid_indices0],)] = src0[valid_indices0]
                # dim>2: include traj dim -- 计算loss时，torchgfn会把所有的traj上的state拼接在一起
                else:
                    tar0[valid_indices0[:-1] +
                         (index0[valid_indices0],)] = src0[valid_indices0]
                self.backward_masks[..., :act_low0] = tar0

                r'''
                # 4) In the final part dealing with binary encoding, a divide-and-co \
                # nquer approach is typically adopted for states 0 and 1. Specifical \
                # ly, the range [one_hot_len*cond0_len: one_hot_len*one_hot_ \
                # seq_len+2*cond1_ndim*cond1_len:2] is designated for instance \
                # s where the value in a given state is 0. Conversely, the range [on \
                # e_hot_ndim*cond0_len+1: one_hot_len*cond0_len+2*binar \
                # y_ndim*cond1_len:2] corresponds to instances where the value  \
                # is 1. For instance, if cond1_ndim*cond1_len equals 3, then t \
                # he action range is [0,5]. Here, 0 and 1 indicate filling the 0th v \
                # alue with 0 and 1, respectively. Similarly, 2 and 3 denote filling \
                # the 1st value with 0 and 1, and 5 and 6 signify filling the 2nd va \
                # lue with 0 and 1.
                '''
                act_low1 = env.one_hot_len*env.cond1_ndim*env.cond1_len
                sta_low1 = env.cond1_ndim*env.cond1_len
                # src: (16, 32) tar: (16, 320) or traj: (72, 16, 32) (72, 16, 320)
                src1 = torch.full_like(
                    self.tensor[..., env.cond0_len:], fill_value=1, dtype=torch.bool, device=self.tensor.device)
                tar1 = torch.zeros_like(
                    self.backward_masks[..., act_low0:], device=self.tensor.device).bool()
                # convert [0, 5, 2, 7] into 0 + (5 + 10) + (2 + 10 + 10) ...
                # 0 is first 10 pos, 5 is second 10 pos, 2 is third 10 pos
                # (bs, len) + (1, len) == 将arange中一行元素依次累加到bs行，实现了上面的0+(5+10)+(2+10+10)
                index1 = self.tensor[..., env.cond0_len:] + (torch.arange(
                    sta_low1, device=self.tensor.device) * env.one_hot_len)[None, ...]
                mask1 = self.tensor[..., env.cond0_len:] >= 0

                # Using advanced indexing instead of the double loop
                # get pos of each nonzero: (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
                valid_indices1 = mask1.nonzero(as_tuple=True)
                # NOTE: only set [0, 5, 2, 7] pos into True, rest is False
                # tar[i,index[i,j]] = src[i,j]
                if len(valid_indices1) == 2:
                    # tuple + same shape tuple = cat(tuple, tuple) --
                    tar1[tuple(valid_indices1[:-1]) +
                         (index1[valid_indices1],)] = src1[valid_indices1]
                # dim>2: include traj dim -- 计算loss时，torchgfn会把所有的traj上的state拼接在一起
                else:
                    tar1[valid_indices1[:-1] +
                         (index1[valid_indices1],)] = src1[valid_indices1]

                self.backward_masks[..., act_low0:] = tar1
                # self.backward_masks[..., act_low0: act_high0:2] = \
                #     self.tensor[..., env.cond0_len:] == 0
                # self.backward_masks[..., act_low0+1: act_high0:2] = \
                #     self.tensor[..., env.cond0_len:] == 1
                # NOTE: for debug
                a = [0]
                # print(f"Finish update mask")

        return DiscreteMSStates

    def is_exit_actions(self, actions: TT["batch_shape"]) -> TT["batch_shape"]:
        return actions == self.n_actions - 1

    # forward one step
    def maskless_step(
        self, states: States, actions: Actions, dones
    ) -> TT["batch_shape", "state_shape", torch.float]:

        # 1) Select the first [0-one_hot_len*cond0_len-1] action to \
        # Update. Due to states.tensor[0:cond0_len-1] is defined as  \
        # one-hot vector, we should first calcuate the index_0 and then    \
        # employ the 'scatter' operation to update state.

        if actions.tensor.shape[0] == 0:
            return states.tensor

        len0 = self.one_hot_len*self.cond0_len
        len1 = len0 + self.one_hot_len*self.cond1_ndim*self.cond1_len

        # NOTE: first 15 items if action < 150:  (16, 1)
        mask_0 = (actions.tensor < len0).squeeze(-1)
        # mask_0 = mask_0 & ~dones
        # index0 for first 15 items (16, 1)
        index_0 = (actions.tensor / self.one_hot_len).long()
        # fmod() 取余, value for first 15 items (16, 1)
        value_0 = torch.fmod(actions.tensor, self.one_hot_len).long()
        # only first 3 is -1 (forward valid), rest is 0 (init zeros)
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            # Set indices to value_0[mask_0] in last dim
            -1, index_0[mask_0], value_0[mask_0]
        )

        # 2) Select the last [one_hot_len*cond0_len:] action to upd \
        # date. We can denote it as one-hot vector with length=2. Besed on \
        # this, we can do the same operation as above.

        mask_1 = (actions.tensor >= len0) & (actions.tensor < len1)
        mask_1 = mask_1.squeeze(-1)
        # mask_1 = mask_1 & ~dones
        index_1 = (((actions.tensor - len0) / self.one_hot_len).long() +
                   self.cond0_len).long()
        value_1 = torch.fmod(actions.tensor - len0, self.one_hot_len).long()
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            # Set indices to value_1[mask1].
            -1, index_1[mask_1], value_1[mask_1]
        )
        return states.tensor

    def maskless_backward_step(
        self, states: States, actions: Actions, dones
    ) -> TT["batch_shape", "state_shape", torch.float]:

        # In this environment, states are represented as n-dimensional ve \
        # ctors where the initial state s0 is an array of -1s. Actions in \
        # the range [0, one_hot_len * cond0_len - 1] change a -1 i \
        # n the state to a value between [0, one_hot_len - 1]. Meanwhile \
        # , actions in the range [one_hot_len * cond0_len, one_hot \
        # _ndim * cond0_len + 2 * cond1_ndim * cond1_len] rep \
        # lace a -1 with a value between [0, 1]. Specifically, an action  \
        # i from the first range updates the state element at index i//o  \
        # ne_hot_ndim with the value i% one_hot_len, whereas an action i \
        # from the second range updates the state element at index (i-one \
        # _hot_ndim * cond0_len)//2 with the value (i-one_hot_len  \
        # * cond0_len)%2. A backward action inquires which index sh \
        # ould revert to -1, which is why the fmod is used to wrap indices.

        # 1) Select the first [0-one_hot_len*cond0_len-1] action to \
        # Update. Due to states.tensor[0:cond0_len-1] is defined as  \
        # one-hot vector, we should first calcuate the index_0 and then    \
        # employ the 'scatter' operation to update state.
        # databases_path, decode, order, cond, ptr, target = info
        if actions.tensor.shape[0] == 0:
            return states.tensor

        len0 = self.one_hot_len*self.cond0_len
        len1 = len0 + self.one_hot_len*self.cond1_ndim*self.cond1_len

        mask_0 = (actions.tensor < len0).squeeze(-1)
        # mask_0 = mask_0 & ~dones
        index_0 = (actions.tensor / self.one_hot_len).long()
        value_0 = -1
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            -1, index_0[mask_0], value_0  # Set indices to -1.
        )

        # 2) Select the last [one_hot_len*cond0_len:] action to upd \
        # date. We can denote it as one-hot vector with length=2. Besed on \
        # this, we can do the same operation as above.

        mask_1 = (actions.tensor >= len0) & (actions.tensor < len1)
        mask_1 = mask_1.squeeze(-1)
        # mask_1 = mask_1 & ~dones
        index_1 = (((actions.tensor - len0) / self.one_hot_len).long() +
                   self.cond0_len).long()
        value_1 = -1
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            -1, index_1[mask_1], value_1  # Set indices to -1.
        )

        return states.tensor

    def log_reward0(self, final_states: DiscreteStates, infos=None) -> TT["batch_shape"]:
        raw_states = final_states.tensor
        canonical = raw_states
        # print("canonical shape = ", canonical.shape)
        # energy is cost model
        # NOTE: Add minus for low GPU latency
        return -self.alpha * self.energy(canonical.float()).clone().detach().view(-1)

    # NOTE: import tlp cost model

    def log_reward(self, final_states: DiscreteStates, infos=None):

        from mlc_dataset.mlc_dataset import restore_embedding
        raw_states = final_states.tensor
        x = raw_states
        # print("canonical shape = ", canonical.shape) # torch.Size([16, 1455])
        # energy is cost model -- tlp
        # NOTE: Add minus for low GPU latency
        self.energy.eval()
        # is_forward, info = infos

        if x.shape[0] == 0:
            return (0, 0)

        # print(f"log reward input x = {x}")
        if torch.all(x == 0):
            print("x is all zeros")
        info = tuple([x])

        info += infos
        # As diff len for each traj, len(features) is diff
        features = restore_embedding(info)
        # print(f"len of features = {len(features)}")
        # print(f"features = {features[0]}")
        # NOTE: This is for speed up predict!
        val_dataloader = SegmentDataloder_new(
            features, shuffle=False, batch_size=x.shape[0]
        )
        pred_results = []
        for batch_data, _ in val_dataloader:
            batch_data = batch_data.to(self.device)
            outputs = self.energy(batch_data)
            if len(pred_results) > 0:
                pred_results += outputs.detach().cpu().tolist()
            else:
                pred_results = outputs.detach().cpu().tolist()

        res = torch.from_numpy(np.array(pred_results)).to(self.device)
        res = res.to(torch.float32)
        # print(f"res = {res}")
        print(
            f"In log_reward() reward = {-self.alpha * res.clone().detach().view(-1)}")
        # return -self.alpha * res.clone().detach().view(-1), res
        # NOTE: TLP 和hardware time 正好相反！！！，将res反向
        return self.alpha * res.clone().detach().view(-1), -res

    # def get_states_indices(self, states: DiscreteStates) -> TT["batch_shape"]:
    #     """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2, then we convert to base 3"""
    #     states_raw = states.tensor
    #     canonical_base = 3 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
    #     return (states_raw + 1).mul(canonical_base).sum(-1).long()

    # def get_terminating_states_indices(
    #     self, states: DiscreteStates
    # ) -> TT["batch_shape"]:
    #     states_raw = states.tensor
    #     canonical_base = 2 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
    #     return (states_raw).mul(canonical_base).sum(-1).long()

    @property
    def n_states(self) -> int:
        return ((1+self.one_hot_len)**self.cond0_len) * (3 ** (self.cond1_len*self.cond1_ndim))

    @property
    def n_terminating_states(self) -> int:
        return (self.one_hot_len**self.cond0_len) * (2 ** (self.cond1_len*self.cond1_ndim))

    @property
    def all_states(self) -> DiscreteStates:
        # This is brute force !
        digits_1 = torch.arange(self.one_hot_len+1, device=self.device)
        digits_2 = torch.arange(3, device=self.device)
        digits = [digits_1] * self.cond0_len + \
            [digits_2] * self.cond1_len*self.cond1_ndim
        all_states = torch.cartesian_prod(*digits) - 1
        return self.States(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        digits_1 = torch.arange(self.one_hot_len, device=self.device)
        digits_2 = torch.arange(2, device=self.device)
        digits = [digits_1] * self.cond0_len + \
            [digits_2] * self.cond1_len*self.cond1_ndim
        all_states = torch.cartesian_prod(*digits)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        true_dist = self.reward(self.terminating_states)
        return true_dist / true_dist.sum()

    @property
    def log_partition(self) -> float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1).item()
