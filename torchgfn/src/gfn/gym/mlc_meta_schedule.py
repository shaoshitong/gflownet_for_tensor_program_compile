from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, cast

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates, States

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
                 binary_ndim=96,
                 one_hot_ndim=10,
                 binary_seq_len=15,
                 one_hot_seq_len=15,
                 energy = None,
                 alpha: float = 1.0,
                 device_str = "cpu",
                 preprocessor_name = "Identity"):
        
        # action's [0,one_hot_ndim*one_hot_seq_len-1]==[0, 15*10-1] represents the choice \
        # of cuda_bind and annotation, while action's [one_hot_ndim*one_hot \
        # _seq_len,one_hot_ndim*one_hot_seq_ len+2*binary_ndim*binary_seq_l \
        # en+1]==[15*10, 15*10+2*15*96-1] represents the choice of sample_perfectile, while action's  \
        # [one_hot_ndim*one_hot_seq_len+2*binary_ndim*binary_seq_len+1] rep \
        # resents the choice of termination condition.
        
        # The dimension of the state is $\mathbb{R}^{one_hot_seq_len+binary \
        # _seq_len*binary_ndim}$==15+15*96, where the first one_hot_seq_len values ar \
        # e in the range of [0,one_hot_ndim-1]==[0, 9] and the inverse binary_seq_len \
        # *binary_ndim values are in the range of [0,1].
        
        self.binary_ndim = binary_ndim
        self.one_hot_ndim = one_hot_ndim
        self.binary_seq_len = binary_seq_len
        self.one_hot_seq_len = one_hot_seq_len
        self.ndim = self.one_hot_seq_len+self.binary_seq_len*self.binary_ndim
        
        s0 = torch.full((self.ndim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((self.ndim,), fill_value=-2, dtype=torch.long, device=torch.device(device_str))
        
        if energy is None:
            energy = IsingModel(
                torch.ones((self.ndim, self.ndim), device=torch.device(device_str))
            )
        self.energy: EnergyFunction = energy
        self.alpha = alpha
        
        n_actions = self.one_hot_ndim*self.one_hot_seq_len+2*self.binary_ndim*self.binary_seq_len + 1
        
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

            # NOTE: get random terminal state
            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> TT["batch_shape", "state_shape", torch.float]:
                one_hot_seq = torch.randint(-1,
                                env.one_hot_ndim-1,
                                batch_shape+(env.one_hot_seq_len,),
                                dtype=torch.long,
                                device=env.device)
                binary_seq  = torch.randint(-1,
                                2,
                                batch_shape+(self.binary_seq_len*self.binary_ndim,),
                                dtype=torch.long,
                                device=env.device)
                total_seq = torch.cat([one_hot_seq,binary_seq],1)
                
                return total_seq
            
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

            def update_masks(self) -> None:
                
                # The following two lines are for typing only.
                
                self.forward_masks = cast(
                    TT["batch_shape", "n_actions", torch.bool],
                    self.forward_masks,
                )
                self.backward_masks = cast(
                    TT["batch_shape", "n_actions - 1", torch.bool],
                    self.backward_masks,
                )

                # The setting of mask is relatively complicated here to realize the \
                # blocking of inappropriate actions during sampling. The algorithm  \
                # is modified from the discrete_edm environment, so most of the pra \
                # ctices are identical to it.
                
                # 1) Both forward and backward masks need to be separated according \
                # to the different parts of the action. This refers to the three di \
                # fferent cases of annotation,sample_perfectile,cuda_bind. In a for \
                # ward mask, state[:one_hot_seq_len] is selectable as long as it is \
                # not -1. Similarly, state[:one_hot_seq_len:] is selectable as long \
                # as it is not -1.

                # NOTE: forward mask is valid position flag (-1) considering according to prim type (in condition)
                self.forward_masks[..., :env.one_hot_ndim*env.one_hot_seq_len] = \
                    (self.tensor[..., :env.one_hot_seq_len] == -1)[...,None].expand(*self.tensor.shape[:-1],
                    env.one_hot_seq_len,env.one_hot_ndim).contiguous().view(*self.tensor.shape[:-1],env.one_hot_seq_len*env.one_hot_ndim)
                
                self.forward_masks[..., env.one_hot_ndim*env.one_hot_seq_len: env.one_hot_ndim*env.one_hot_seq_len+2*env.binary_ndim*env.binary_seq_len] = \
                    (self.tensor[..., env.one_hot_seq_len:] == -1)[...,None].expand(*self.tensor.shape[:-1],
                    env.binary_ndim*env.binary_seq_len,2).contiguous().view(*self.tensor.shape[:-1],env.binary_ndim*env.binary_seq_len*2)
                
                # 2) The last action is not a terminal state as long as one of the e \
                # xistent states is -1.
                
                self.forward_masks[..., -1] = torch.all(self.tensor != -1, dim=-1)
                
                # 3) The case of backward's mask is more complex, and we need to emp \
                # loy torch.scatter to insert executable markers at the appropriate  \
                # index positions. Specifically, when there is a value other than -1 \
                # in the previous one_hot_seq_len values, then there is an action th \
                # at cannot be selected. Assuming that the action corresponds to pos \
                # ition K of one_hot_seq_len, then it is computed as [K*one_hot_ndim \
                # +action_value].

                # NOTE: backward mask is valid position flag (not -1)

                src = torch.full_like(self.tensor[..., :env.one_hot_seq_len], fill_value=1, dtype=torch.bool, device=self.tensor.device)
                tar = torch.zeros_like(self.backward_masks[..., :env.one_hot_ndim*env.one_hot_seq_len], device=self.tensor.device).bool()
                # convert [0, 5, 2, 7] into 0 + (5 + 10) + (2 + 10 + 10) ...
                # 0 is first 10 pos, 5 is second 10 pos, 2 is third 10 pos 
                index = self.tensor[..., :env.one_hot_seq_len] + (torch.arange(env.one_hot_seq_len, device=self.tensor.device) * env.one_hot_ndim)[None, ...]
                mask = self.tensor[..., :env.one_hot_seq_len] >= 0

                # Using advanced indexing instead of the double loop
                valid_indices = mask.nonzero(as_tuple=True)
                if len(valid_indices) == 2:
                    tar[tuple(valid_indices[:-1]) + (index[valid_indices],)] = src[valid_indices]
                else:
                    tar[valid_indices[:-1] + (index[valid_indices],)] = src[valid_indices]       
                self.backward_masks[..., :env.one_hot_ndim*env.one_hot_seq_len] = tar

                # 4) In the final part dealing with binary encoding, a divide-and-co \
                # nquer approach is typically adopted for states 0 and 1. Specifical \
                # ly, the range [one_hot_ndim*one_hot_seq_len: one_hot_ndim*one_hot_ \
                # seq_len+2*binary_ndim*binary_seq_len:2] is designated for instance \
                # s where the value in a given state is 0. Conversely, the range [on \
                # e_hot_ndim*one_hot_seq_len+1: one_hot_ndim*one_hot_seq_len+2*binar \
                # y_ndim*binary_seq_len:2] corresponds to instances where the value  \
                # is 1. For instance, if binary_ndim*binary_seq_len equals 3, then t \
                # he action range is [0,5]. Here, 0 and 1 indicate filling the 0th v \
                # alue with 0 and 1, respectively. Similarly, 2 and 3 denote filling \
                # the 1st value with 0 and 1, and 5 and 6 signify filling the 2nd va \
                # lue with 0 and 1.
                
                self.backward_masks[..., env.one_hot_ndim*env.one_hot_seq_len: env.one_hot_ndim*env.one_hot_seq_len+2*env.binary_ndim*env.binary_seq_len:2] = \
                    self.tensor[..., env.one_hot_seq_len:] == 0      
                self.backward_masks[..., env.one_hot_ndim*env.one_hot_seq_len+1: env.one_hot_ndim*env.one_hot_seq_len+2*env.binary_ndim*env.binary_seq_len:2] = \
                    self.tensor[..., env.one_hot_seq_len:] == 1    


        return DiscreteMSStates
 
    def is_exit_actions(self, actions: TT["batch_shape"]) -> TT["batch_shape"]:
        return actions == self.n_actions - 1
    
    # forward one step
    def maskless_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        
        # 1) Select the first [0-one_hot_ndim*one_hot_seq_len-1] action to \
        # Update. Due to states.tensor[0:one_hot_seq_len-1] is defined as  \
        # one-hot vector, we should first calcuate the index_0 and then    \
        # employ the 'scatter' operation to update state.
        
        mask_0 = (actions.tensor < self.one_hot_ndim*self.one_hot_seq_len).squeeze(-1)
        index_0 = (actions.tensor / self.one_hot_ndim).long()
        # fmod() 取余
        value_0 = torch.fmod(actions.tensor, self.one_hot_ndim).long()
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            -1, index_0[mask_0], value_0[mask_0]  # Set indices to value_0[mask_0].
        )
        
        # 2) Select the last [one_hot_ndim*one_hot_seq_len:] action to upd \
        # date. We can denote it as one-hot vector with length=2. Besed on \
        # this, we can do the same operation as above.
        
        mask_1 = (actions.tensor >= self.one_hot_ndim*self.one_hot_seq_len) & \
            (actions.tensor < self.one_hot_ndim*self.one_hot_seq_len+2*self.binary_ndim*self.binary_seq_len)
        mask_1 = mask_1.squeeze(-1)
        index_1 = (((actions.tensor - self.one_hot_ndim*self.one_hot_seq_len) / 2).int() + self.one_hot_seq_len).long()
        value_1 = torch.fmod(actions.tensor - self.one_hot_ndim*self.one_hot_seq_len , 2).long()
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            -1, index_1[mask_1], value_1[mask_1]  # Set indices to value_1[mask1].
        )
        return states.tensor
    
    def maskless_backward_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        
        # In this environment, states are represented as n-dimensional ve \
        # ctors where the initial state s0 is an array of -1s. Actions in \
        # the range [0, one_hot_ndim * one_hot_seq_len - 1] change a -1 i \
        # n the state to a value between [0, one_hot_ndim - 1]. Meanwhile \
        # , actions in the range [one_hot_ndim * one_hot_seq_len, one_hot \
        # _ndim * one_hot_seq_len + 2 * binary_ndim * binary_seq_len] rep \
        # lace a -1 with a value between [0, 1]. Specifically, an action  \
        # i from the first range updates the state element at index i//o  \
        # ne_hot_ndim with the value i% one_hot_ndim, whereas an action i \
        # from the second range updates the state element at index (i-one \
        # _hot_ndim * one_hot_seq_len)//2 with the value (i-one_hot_ndim  \
        # * one_hot_seq_len)%2. A backward action inquires which index sh \
        # ould revert to -1, which is why the fmod is used to wrap indices.

        # 1) Select the first [0-one_hot_ndim*one_hot_seq_len-1] action to \
        # Update. Due to states.tensor[0:one_hot_seq_len-1] is defined as  \
        # one-hot vector, we should first calcuate the index_0 and then    \
        # employ the 'scatter' operation to update state.
        
        mask_0 = (actions.tensor < self.one_hot_ndim*self.one_hot_seq_len).squeeze(-1)
        index_0 = (actions.tensor / self.one_hot_ndim).long()
        value_0 = -1
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            -1, index_0[mask_0], value_0  # Set indices to -1.
        )
        
        # 2) Select the last [one_hot_ndim*one_hot_seq_len:] action to upd \
        # date. We can denote it as one-hot vector with length=2. Besed on \
        # this, we can do the same operation as above.
        
        mask_1 = (actions.tensor >= self.one_hot_ndim*self.one_hot_seq_len) & \
            (actions.tensor < self.one_hot_ndim*self.one_hot_seq_len+2*self.binary_ndim*self.binary_seq_len)
        mask_1 = mask_1.squeeze(-1)
        index_1 = (((actions.tensor - self.one_hot_ndim*self.one_hot_seq_len) / 2).int() + self.one_hot_seq_len).long()
        value_1 = -1
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            -1, index_1[mask_1], value_1  # Set indices to -1.
        )
        
        return states.tensor
    
    def log_reward(self, final_states: DiscreteStates) -> TT["batch_shape"]:
        raw_states = final_states.tensor
        canonical = raw_states
        # energy is cost model
        return self.alpha * self.energy(canonical.float()).clone().detach().view(-1)

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
        return ((1+self.one_hot_ndim)**self.one_hot_seq_len) * (3 ** (self.binary_seq_len*self.binary_ndim))

    @property
    def n_terminating_states(self) -> int:
        return (self.one_hot_ndim**self.one_hot_seq_len) * (2 ** (self.binary_seq_len*self.binary_ndim))

    @property
    def all_states(self) -> DiscreteStates:
        # This is brute force !
        digits_1 = torch.arange(self.one_hot_ndim+1, device=self.device)
        digits_2 = torch.arange(3, device=self.device)
        digits = [digits_1] * self.one_hot_seq_len + [digits_2] * self.binary_seq_len*self.binary_ndim
        all_states = torch.cartesian_prod(*digits) - 1
        return self.States(all_states)
    
    @property
    def terminating_states(self) -> DiscreteStates:
        digits_1 = torch.arange(self.one_hot_ndim, device=self.device)
        digits_2 = torch.arange(2, device=self.device)
        digits = [digits_1] * self.one_hot_seq_len + [digits_2] * self.binary_seq_len*self.binary_ndim
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
