from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, cast

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from src.gfn.actions import Actions
from src.gfn.env import DiscreteEnv
from src.gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from src.gfn.states import DiscreteStates, States


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


class DiscreteEBM(DiscreteEnv):
    """Environment for discrete energy-based models, based on https://arxiv.org/pdf/2202.01361.pdf"""

    def __init__(
        self,
        ndim: int,
        energy = None,
        alpha: float = 1.0,
        device_str = "cpu",
        preprocessor_name = "Identity",
    ):
        """Discrete EBM environment.

        Args:
            ndim (int, optional): dimension D of the sampling space {0, 1}^D.
            energy (EnergyFunction): energy function of the EBM. Defaults to None. If None, the Ising model with Identity matrix is used.
            alpha (float, optional): interaction strength the EBM. Defaults to 1.0.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        # -1 represent init state, 2 represent finish state
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), 2, dtype=torch.long, device=torch.device(device_str))

        if energy is None:
            energy = IsingModel(
                torch.ones((ndim, ndim), device=torch.device(device_str))
            )
        self.energy: EnergyFunction = energy
        self.alpha = alpha
        
        # Each state has binary value
        n_actions = 2 * ndim + 1
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2 * ndim - 1] corresponds to replacing s[i - ndim] with 1

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_dim=ndim)
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

        class DiscreteEBMStates(DiscreteStates):
            state_shape = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions
            device = env.device

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> TT["batch_shape", "state_shape", torch.float]:
                # ret tensor filled with random integers between [low, high) = [-1, 2)
                return torch.randint(
                    -1,
                    2,
                    # (bs, ) + (ndim, ) ==> (bs, ndim)
                    batch_shape + (env.ndim,),
                    dtype=torch.long,
                    device=env.device,
                )

            def make_masks(
                self,
            ) -> Tuple[
                TT["batch_shape", "n_actions", torch.bool],
                TT["batch_shape", "n_actions - 1", torch.bool],
            ]:
                forward_masks = torch.zeros(
                    self.batch_shape + (env.n_actions,), # (bs, n_act)
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
                # The following two lines are for casting type only.
                # forward_mask shape is (batch_shape, n_actions), type is bool
                self.forward_masks = cast(
                    TT["batch_shape", "n_actions", torch.bool],
                    self.forward_masks,
                )
                self.backward_masks = cast(
                    TT["batch_shape", "n_actions - 1", torch.bool],
                    self.backward_masks,
                )

                # mask mark where is valid pos for forward & backward
                # forward pos is val == -1 for first & second part, but final action is all val != -1
                # backward first part is val == 0, second part is val == 1
                # ... 
                self.forward_masks[..., : env.ndim] = self.tensor == -1
                self.forward_masks[..., env.ndim : 2 * env.ndim] = self.tensor == -1
                # torch.all(input): test if all eles in input evaluate to True
                self.forward_masks[..., -1] = torch.all(self.tensor != -1, dim=-1)
                
                self.backward_masks[..., : env.ndim] = self.tensor == 0
                self.backward_masks[..., env.ndim : 2 * env.ndim] = self.tensor == 1

        return DiscreteEBMStates
    # check if in final action
    def is_exit_actions(self, actions: TT["batch_shape"]) -> TT["batch_shape"]:
        return actions == self.n_actions - 1
    # 针对 batch 根据action 将 state 某一个位置 设为0/1
    def maskless_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        # First, we select that actions that replace a -1 with a 0.
        # Remove singleton dimension for broadcasting.

        # First make mask for action replacing -1 with 0
        # mask_0 == [1, 0, 1, ..] mark if action < ndim
        mask_0 = (actions.tensor < self.ndim).squeeze(-1)
        # print(mask_0.shape)
        # print(states.tensor.shape,actions.tensor.shape)
        # torch.Size([16, 784]) torch.Size([16, 1])                                                                                                                                                 
        # torch.Size([16])
        
        # dim=-1 represent last dim -- innermost dim [[[-1, -1, -1]]]
        # Based on action & mask_0, set state into 0 -- final pos is pointed out action
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            -1, actions.tensor[mask_0], 0  # Set indices to 0.
        )
        # Then, we select that actions that replace a -1 with a 1.
        # mask_1 == [0, 1, 0, ..] mark if action >= ndim & action < 2*ndim
        mask_1 = (
            (actions.tensor >= self.ndim) & (actions.tensor < 2 * self.ndim)
        ).squeeze(
            -1
        )  # Remove singleton dimension for broadcasting.
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            # map [ndim, 2*ndim-1] into [0, ndim-1]
            -1, (actions.tensor[mask_1] - self.ndim), 1  # Set indices to 1.
        )
        return states.tensor

    def maskless_backward_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        # In this env, states are n-dim vectors. s0 is empty (represented as -1),
        # so s0=[-1, -1, ..., -1], each action is replacing a -1 with either a
        # 0 or 1. Action i in [0, ndim-1] is replacing s[i] with 0, whereas
        # action i in [ndim, 2*ndim-1] corresponds to replacing s[i - ndim] with 1.
        # A backward action asks "what index should be set back to -1", hence the fmod
        # to enable wrapping of indices.
        # NOTE: As action \in [0, ndim-1] will set 0 in [0, ndim-1] as -1, 
        # action \in [ndim, 2*ndim-1] will set 1 in [0, ndim-1] as -1 -- i & i+ndim map same pos i
        # Use fmod() as operator tensor
        # >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
        # tensor([-1., -0., -1.,  1.,  0.,  1.])
        return states.tensor.scatter(
            -1,
            actions.tensor.fmod(self.ndim),
            -1,
        )

    def log_reward(self, final_states: DiscreteStates) -> TT["batch_shape"]:
        raw_states = final_states.tensor
        canonical = raw_states
        # NOTE: modify for detach().view(-1) -- use energy() calculate reward
        return -self.alpha * self.energy(canonical.float()).clone().detach().view(-1)

    def get_states_indices(self, states: DiscreteStates) -> TT["batch_shape"]:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2, then we convert to base 3"""
        states_raw = states.tensor
        canonical_base = 3 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        # sum(-1): along last dim compute sum
        return (states_raw + 1).mul(canonical_base).sum(-1).long()

    def get_terminating_states_indices(
        self, states: DiscreteStates
    ) -> TT["batch_shape"]:
        states_raw = states.tensor
        canonical_base = 2 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        return (states_raw).mul(canonical_base).sum(-1).long()

    @property
    def n_states(self) -> int:
        return 3**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return 2**self.ndim

    @property
    def all_states(self) -> DiscreteStates:
        # This is brute force !
        digits = torch.arange(3, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        all_states = all_states - 1
        return self.States(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        digits = torch.arange(2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        true_dist = self.reward(self.terminating_states)
        return true_dist / true_dist.sum()

    @property
    def log_partition(self) -> float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1).item()