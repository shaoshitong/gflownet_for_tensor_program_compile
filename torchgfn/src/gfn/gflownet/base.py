from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from src.gfn.containers import Trajectories
from src.gfn.containers.base import Container
from src.gfn.env import Env
from src.gfn.modules import GFNModule
from src.gfn.samplers import Sampler
from src.gfn.states import States

TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, Tuple[States, ...]]
)


class GFlowNet(ABC, nn.Module, Generic[TrainingSampleType]):
    """Abstract Base Class for GFlowNets.

    A formal definition of GFlowNets is given in Sec. 3 of [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266).
    """

    @abstractmethod
    def sample_trajectories(self, env: Env, n_samples: int) -> Trajectories:
        """Sample a specific number of complete trajectories.

        Args:
            env: the environment to sample trajectories from.
            n_samples: number of trajectories to be sampled.
        Returns:
            Trajectories: sampled trajectories object.
        """

    def sample_terminating_states(self, env: Env, n_samples: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            env: the environment to sample terminating states from.
            n_samples: number of terminating states to be sampled.
        Returns:
            States: sampled terminating states object.
        """
        trajectories = self.sample_trajectories(env, n_samples)
        return trajectories.last_states

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories) -> TrainingSampleType:
        """Converts trajectories to training samples. The type depends on the GFlowNet."""

    @abstractmethod
    def loss(self, env: Env, training_objects):
        """Computes the loss given the training objects."""


class PFBasedGFlowNet(GFlowNet):
    r"""Base class for gflownets that explicitly uses $P_F$.

    Attributes:
        pf: GFNModule
        pb: GFNModule
    """

    def __init__(self, pf: GFNModule, pb: GFNModule, on_policy: bool = False):
        super().__init__()
        self.pf = pf
        self.pb = pb
        self.on_policy = on_policy

    def sample_trajectories(self, env: Env, n_samples: int, info=None) -> Trajectories:
        sampler = Sampler(estimator=self.pf)
        trajectories = sampler.sample_trajectories(
            env, n_trajectories=n_samples, info=info)
        return trajectories


class TrajectoryBasedGFlowNet(PFBasedGFlowNet):
    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
    ) -> Tuple[
        TT["max_length", "n_trajectories", torch.float],
        TT["max_length", "n_trajectories", torch.float],
    ]:
        r"""Evaluates log probs for each transition in each trajectory in the batch.

        More specifically it evaluates $\log P_F (s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        Useful when the policy used to sample the trajectories is different from
        the one used to evaluate the loss. Otherwise we can use the logprobs directly
        from the trajectories.

        Args:
            trajectories: Trajectories to evaluate.
            fill_value: Value to use for invalid states (i.e. $s_f$ that is added to
                shorter trajectories).

        Returns: A tuple of float tensors of shape (max_length, n_trajectories) containing
            the log_pf and log_pb for each action in each trajectory. The first one can be None.

        Raises:
            ValueError: if the trajectories are backward.
            AssertionError: when actions and states dimensions mismatch.
        """
        # fill value is the value used for invalid states (sink state usually)
        # NOTE: need for double check! -- refer gflownet/tb_mle_learnedpb_loss
        if trajectories.is_backward:
            # traj: (73, 16) -- step by step all batch(16) become all True
            # no init: is_initial_state: only last bs(16) is init state, all not dummy
            # with init: is_initial_state: last #72 all is True, #71 only one False, #70 only one False
            # valid_state: (727, 1455) valid_action: (727, 1) traj.state: (73, 16, 1455) traj.action: (72, 16)
            valid_states = trajectories.states[~trajectories.states.is_initial_state]
            valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

            # uncomment next line for debugging
            # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions.is_dummy)
            if valid_states.batch_shape != tuple(valid_actions.batch_shape):
                raise AssertionError(
                    "Something wrong happening with log_pf evaluations")

            if self.on_policy:
                log_pb_trajectories = trajectories.log_probs
            else:
                module_output = self.pb(valid_states)
                valid_log_pb_actions = self.pb.to_probability_distribution(
                    valid_states, module_output
                ).log_prob(valid_actions.tensor)
                # NOTE: fill_value!
                log_pb_trajectories = torch.full_like(
                    trajectories.actions.tensor[..., 0],
                    fill_value=fill_value,
                    dtype=torch.float,
                )  # compute log_pb of all state backward one step along traj
                log_pb_trajectories[~trajectories.actions.is_dummy] = valid_log_pb_actions
            # # create repeated s0 with (batch, s0.shape)
            # source_states_tensor = valid_states.s0.repeat(
            #     *valid_states.batch_shape, *  # 727
            #     ((1,) * len(valid_states.state_shape))  # 1
            # )
            # out = valid_states.tensor == source_states_tensor
            # state_ndim = len(valid_states.state_shape)
            # for _ in range(state_ndim):
            #     out = out.any(dim=-1)
            # # out: if is -1 init value
            # is_sink_state = ~out

            def is_finish(a, b):
                res = []
                for i in range(a.shape[0]):
                    flag = False
                    for j in range(b.shape[0]):
                        if torch.all(a[i] == b[j]):
                            flag = True
                            break
                    res.append(flag)
                return torch.tensor(res)
            # valid_states: (727, 1455) backward_state: (16, 1455)
            # is_sink_state = is_finish(valid_states.tensor, trajectories.start_state.tensor)
            # NOTE: is_sink_state: only first 16 is True
            # NOTE: is_init_state: only duplicate s0 is True, except first s0
            is_init_state = torch.zeros_like(
                trajectories.states.is_initial_state)
            src_tensor = trajectories.states.tensor
            s0 = valid_states.s0
            n, m, _ = trajectories.states.tensor.shape
            for i in range(n):
                id = n-i-1
                is_exit = 0
                flag = 0
                mark = []
                for j in range(m):
                    flag = 0
                    # remove all s0 except last s0 in each traj
                    if torch.all(s0 == src_tensor[id][j]) and \
                            torch.all(s0 == src_tensor[id-1][j]):
                        flag = 1
                    is_init_state[id][j] = flag
                    mark.append(flag)
                if (torch.tensor(mark) == 0).all():
                    break

            non_sink_valid_states = trajectories.states[~is_init_state]
            start_tensor = trajectories.start_state.tensor
            cur_tensor = non_sink_valid_states.tensor
            # NOTE: only first 16 is False
            is_sink_state = is_finish(cur_tensor, start_tensor)
            non_sink_valid_states = non_sink_valid_states[~is_sink_state]
            # source_actions_tensor = torch.zeros_like(
            #     trajectories.actions.tensor)
            # source_actions_tensor[-1, ...] = 1
            # is_initial_actions = source_actions_tensor.bool()
            # is_initial_actions = is_initial_actions[~trajectories.actions.is_dummy]
            # # is_initial_actions only [725, 0] & [726, 0] is True
            # # only last 16 is True
            # non_exit_valid_actions = valid_actions[~is_initial_actions]
            # not need mask action
            non_exit_valid_actions = valid_actions
            # # NOTE: remove first n, mark invalid state!
            # nn = non_sink_valid_states.tensor.shape[0] - \
            #     non_exit_valid_actions.tensor.shape[0]
            # non_sink_valid_states = non_sink_valid_states[nn:]

            # states: (727, 1455) action: (725)
            module_output = self.pf(non_sink_valid_states)
            # NOTE: check if -inf
            valid_log_pf_actions = self.pf.to_probability_distribution(
                non_sink_valid_states, module_output
            ).log_prob(non_exit_valid_actions.tensor)

            log_pf_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.float,
            )
            # log_pf_trajectories_slice = torch.full_like(
            #     valid_actions.tensor[...,
            #                          0], fill_value=fill_value, dtype=torch.float
            # )
            # log_pf_trajectories_slice[~trajectories.actions.is_dummy] = valid_log_pf_actions
            # log_pf_trajectories_slice[~is_initial_actions.squeeze(
            #     -1)] = valid_log_pf_actions
            log_pf_trajectories[~trajectories.actions.is_dummy] = valid_log_pf_actions

            return log_pf_trajectories, log_pb_trajectories
        else:
            # NOTE: Forward traj -- only last 16 is sink state, all not dummy action
            # states (1457, 16, 1455)  action (1456, 16, 1)
            valid_states = trajectories.states[~trajectories.states.is_sink_state]
            valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

            # uncomment next line for debugging
            # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions.is_dummy)

            if valid_states.batch_shape != tuple(valid_actions.batch_shape):
                raise AssertionError(
                    "Something wrong happening with log_pf evaluations")
            # forward_masks
            if self.on_policy:
                log_pf_trajectories = trajectories.log_probs
            else:
                module_output = self.pf(valid_states)
                # NOTE: first 16 states is -inf
                valid_log_pf_actions = self.pf.to_probability_distribution(
                    valid_states, module_output
                ).log_prob(valid_actions.tensor)
                log_pf_trajectories = torch.full_like(
                    trajectories.actions.tensor[..., 0],
                    fill_value=fill_value,
                    dtype=torch.float,
                )
                log_pf_trajectories[~trajectories.actions.is_dummy] = valid_log_pf_actions
            # NOTE: states/actions (80, 1455) (64, 1) -- (80, 1455) (80, 1)
            # is_initial_state: only first 16 is True
            non_initial_valid_states = valid_states[~valid_states.is_initial_state]

            # is_exit: only last 16 is True
            non_exit_valid_actions = valid_actions[~valid_actions.is_exit]
            # NOTE: remove first n, mark invalid state!
            nn = non_initial_valid_states.tensor.shape[0] - \
                non_exit_valid_actions.tensor.shape[0]
            non_initial_valid_states = non_initial_valid_states[nn:]
            # NOTE: check if last n is init -- wrong!
            # non_initial_valid_states = non_initial_valid_states[:-nn]

            module_output = self.pb(non_initial_valid_states)
            # valid_log_pb_actions: (23280)
            valid_log_pb_actions = self.pb.to_probability_distribution(
                non_initial_valid_states, module_output
            ).log_prob(non_exit_valid_actions.tensor)
            # log_pb_trajectories: (1456, 16) -- (23296)
            log_pb_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.float,
            )
            log_pb_trajectories_slice = torch.full_like(
                valid_actions.tensor[...,
                                     0], fill_value=fill_value, dtype=torch.float
            )
            # log_pb_trajectories_slice: (23296)
            log_pb_trajectories_slice[~valid_actions.is_exit] = valid_log_pb_actions
            log_pb_trajectories[~trajectories.actions.is_dummy] = log_pb_trajectories_slice

            return log_pf_trajectories, log_pb_trajectories

    def get_trajectories_scores(
        self, trajectories: Trajectories
    ) -> Tuple[
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
    ]:
        """Given a batch of trajectories, calculate forward & backward policy scores."""
        # log pf and log pb along traj: 15+15*96 + 2(special state)
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories)

        assert log_pf_trajectories is not None
        total_log_pf_trajectories = log_pf_trajectories.sum(dim=0)
        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0)
        # reward info for supervising
        # print(f"In get_traj_score, reward = {trajectories.log_rewards}")
        # log_rewards = trajectories.log_rewards.clamp_min(
        #     self.log_reward_clip_min)  # type: ignore
        # log_rewards = trajectories.log_rewards.clamp_min(
        #     -5000)  # type: ignore
        log_rewards = trajectories.log_rewards
        # print(f"In get_traj_score, reward = {log_rewards}")

        if torch.any(torch.isinf(total_log_pf_trajectories)) or torch.any(
            torch.isinf(total_log_pb_trajectories)
        ):
            raise ValueError("Infinite logprobs found")
        return (
            total_log_pf_trajectories,
            total_log_pb_trajectories,
            # ret pf - pb - logR. caller will add logZ
            total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        )

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        return trajectories
