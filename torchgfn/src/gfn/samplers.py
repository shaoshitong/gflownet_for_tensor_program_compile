from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType as TT

from src.gfn.actions import Actions
from src.gfn.containers import Trajectories
from src.gfn.env import Env
from src.gfn.modules import GFNModule
from src.gfn.states import States
from src.gfn.states import DiscreteStates, States


class Sampler:
    """`Sampler is a container for a PolicyEstimator.

    Can be used to either sample individual actions, sample trajectories from $s_0$,
    or complete a batch of partially-completed trajectories from a given batch states.

    Attributes:
        estimator: the submitted PolicyEstimator.
        probability_distribution_kwargs: keyword arguments to be passed to the `to_probability_distribution`
             method of the estimator. For example, for DiscretePolicyEstimators, the kwargs can contain
             the `temperature` parameter, `epsilon`, and `sf_bias`.
    """

    def __init__(
        self,
        estimator: GFNModule,
        **probability_distribution_kwargs: Optional[dict],
    ) -> None:
        self.estimator = estimator
        self.probability_distribution_kwargs = probability_distribution_kwargs

    def sample_actions(
        self, env: Env, states: States, info
    ) -> Tuple[Actions, TT["batch_shape", torch.float]]:
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
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

        # NOTE: unable stack for unequal size [3, 24] & [4, 24], only use list
        # cond0 = torch.stack(cond0, 0)
        # cond1 = torch.stack(cond1, 0)

        def check(res, cond0, cond1, mask):
            res = res.squeeze(-1)
            # # NOTE: DEBUG
            # res[:6] = 9

            is_valid = torch.ones_like(res)  # is valid
            old_len0 = old_len1 = 0
            len0 = 150
            # NOTE: first 15 items if action < 150:  (16, 1)
            mask_0 = (res < len0).squeeze(-1)
            # index0 for first 15 items (16, 1)
            index_0 = (res / 10).long()
            for id in range(len(cond0)):  # (16)
                if cond0[id].shape[0] > 0:
                    old_len0 = cond0[id][..., 3].to("cuda")  # (3, 24)
                    n = len(cond0)
                    m = old_len0.shape[0]
                    flag = True
                    # NOTE: must satisfy all condition
                    for i in range(m):
                        lw = 10*i
                        hi = 10*i + old_len0[i].item()
                        ma = 10 * (i+1)
                        if res[id] in range(int(hi), ma):
                            flag = False
                    is_valid[id] = flag
                    # lw = hi = torch.tensor([10*i for j in range(n)]).to("cuda")
                    # hi = torch.add(hi, old_len0[:, i].squeeze(-1)).to("cuda")
                    # ma = torch.tensor([10*(i+1) for j in range(n)]).to("cuda")
                    # 0 <= res <= 10
                    # c1 = torch.logical_and(
                    #     torch.le(lw, res[id]), torch.lt(res[id], ma))
                    # # hi <= res <= 10
                    # c2 = torch.logical_and(c1, torch.le(hi, res[id]))
                    # is_valid = torch.logical_and(
                    #     is_valid, torch.logical_not(c2))
                # TODO: satisfy cond1
                # if cond1[id].shape[0] > 0:
                #     old_len1 = cond1[id][..., 0].to("cuda") # (3, 34)
                #     m = old_len1.shape[0]
                #     flag = True
                #     # NOTE: must satisfy all condition
                #     for i in range(m):
                #         lw = 10*i
                #         hi = 10*i + old_len0[i].item()
                #         ma = 10 * (i+1)
                #         if res[id] in range(int(hi), ma):
                #             flag = False
                #     is_valid[id] = flag

            # if (is_valid == 1).all():
            #     return -1
            # else:
            is_valid = torch.logical_or(is_valid, mask.to("cuda"))
            pos = torch.tensor([i for i in range(res.shape[0])])
            return pos[is_valid == 0]


        module_output = self.estimator(states)
        # distribution for sample
        dist = self.estimator.to_probability_distribution(
            states, module_output, **self.probability_distribution_kwargs
        )

        with torch.no_grad():
            res = dist.sample()
            actions = res

            # # NOTE: mark pre valid value avoiding repeat sample
            # mask = torch.zeros(res.shape[0])
            # new_mask = mask
            # pos = check(res, cond0, cond1, new_mask)
            # while (pos.shape[0] > 0):
            #     res0 = dist.sample()
            #     actions[pos] = res0[pos]
            #     mask = torch.zeros(res.shape[0])
            #     mask[pos] = 1
            #     mask = torch.logical_not(mask)
            #     new_mask = torch.logical_or(new_mask, mask)
            #     pos = check(res0, cond0, cond1, new_mask)

        log_probs = dist.log_prob(actions)
        if torch.any(torch.isinf(log_probs)):
            raise RuntimeError(
                "Log probabilities are inf. This should not happen.")

        return env.Actions(actions), log_probs

    def sample_backward_actions(
        self, env: Env, states: States, info
    ) -> Tuple[Actions, TT["batch_shape", torch.float]]:
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """

        module_output = self.estimator(states)
        # distribution for sample
        dist = self.estimator.to_probability_distribution(
            states, module_output, **self.probability_distribution_kwargs
        )
        # mask invalid action
        with torch.no_grad():
            res = dist.sample()
            actions = res

        log_probs = dist.log_prob(actions)
        if torch.any(torch.isinf(log_probs)):
            raise RuntimeError(
                "Log probabilities are inf. This should not happen.")

        return env.Actions(actions), log_probs

    def init_cond(self, states, info, val):
        # NOTE: set padding into zeros
        databases_paths, decodes, orders, conds, ptrs, target = info
        for i in range(states.tensor.shape[0]):
            state = states.tensor[i]
            database_path, decode, order, cond, ptr = \
                databases_paths[i], decodes[i], orders[i], conds[i], ptrs[i]
            # (3) (24) (0) (0)
            cond_x0, cond_y0, cond_x1, cond_y1, max_len, emb0_x, emb1_x = decode
            len0 = 15
            fac = 32
            len1 = 1455

            # TODO: set invalid part(cond0 padding into 15) into zero
            st0 = cond_x0.item()
            # print(f"cond0 init {state[st0]} in {st0}:{len0} into {val}")
            state[st0:len0] = val
            # TODO: set invalid part(cond1 padding into 15) into zero
            st1 = len0 + cond_x1.item() * fac
            # print(f"cond1 init {state[st1]} in {st1}:{state.shape[0]} into {val}")
            state[st1:] = val
            states.tensor[i] = state

            ex_cond0, ex_cond1 = torch.split(
                cond, split_size_or_sections=15, dim=0)  # split along 1dim

            t0, _ = torch.split(
                ex_cond0, [cond_x0.item(), 15-cond_x0.item()], 0)
            t1, _ = torch.split(
                ex_cond1, [cond_x1.item(), 15-cond_x1.item()], 0)

            t0, _ = torch.split(
                t0, [cond_y0.item(), 34-cond_y0.item()], 1)
            t1, _ = torch.split(
                t1, [cond_y1.item(), 34-cond_y1.item()], 1)

            # NOTE: init zero for greater than len(factors)
            import numpy as np
            if t1.shape[0] > 0:
                m = t1.shape[0]
                for j in range(m):
                    tt = t1[j].numpy()  # (4, 34)
                    factors = tt[2:]
                    ed = np.argmin(factors != 0)
                    factors = factors[:ed]
                    lenf = len(factors)
                    len0 = 15
                    delta = 32
                    state = states.tensor[i]
                    st = len0 + j*delta + lenf
                    ed = len0 + (j+1)*delta
                    # print(f"tile len cond1 init {state[st]} in {st}:{ed} into {val}")
                    state[st:ed] = val
                    states.tensor[i] = state
            # # NOTE: must set is_initial_state otherwise state=(80, 1455) action=(64, 1)
            # states.is_initial_state = True -- syntax error, solve in base.py
            # NOTE: must add update_masks(), otherwise log_pf will be -inf
        if isinstance(states, DiscreteStates):
            states.update_masks(info)
        from copy import deepcopy
        return deepcopy(states)

    def sample_trajectories(
        self,
        env: Env,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
        info=None,
    ) -> Trajectories:
        """Sample trajectories sequentially.

        Args:
            env: The environment to sample trajectories from.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            n_trajectories: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.

        Returns: A Trajectories object representing the batch of sampled trajectories.

        Raises:
            AssertionError: When both states and n_trajectories are specified.
            AssertionError: When states are not linear.
        """
        backward_state = None
        start_state = None
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = env.reset(batch_shape=(n_trajectories,))
            states = self.init_cond(states, info, 0)

        else:
            # NOTE: first get original states!
            from copy import deepcopy
            backward_state = deepcopy(states)

            states = self.init_cond(states, info, -1)
            # NOTE: get start state in backward
            start_state = states
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.tensor.device
        # finish state: is backward, state is initial state, otherwise sink state
        dones = (
            states.is_initial_state
            if self.estimator.is_backward
            else states.is_sink_state
        )
        # states in traj
        trajectories_states: List[TT["n_trajectories", "state_shape", torch.float]] = [
            states.tensor
        ]
        # action in traj
        trajectories_actions: List[TT["n_trajectories", torch.long]] = []
        # log prob of action in traj
        trajectories_logprobs: List[TT["n_trajectories", torch.float]] = []
        #
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )
        features = None
        step = 0
        # condition is True if done not all True -- not reach finish state
        while not all(dones):
            # create action of dummy action with shape is (n_traj, )
            actions = env.Actions.make_dummy_actions(
                batch_shape=(n_trajectories,))
            # log prob of action
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            # NOTE: step 1 -- Sample action & log prob
            if self.estimator.is_backward:
                valid_actions, actions_log_probs = self.sample_backward_actions(
                    env, states[~dones], info)
            else:
                valid_actions, actions_log_probs = self.sample_actions(
                    env, states[~dones], info)

            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            # add new action & log prob -- 5th all is exit for actions
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]
            # NOTE: step 2 -- apply action to state
            # if backward sampler, apply action to state, get new state
            if self.estimator.is_backward:
                new_states = env.backward_step(states, actions, dones, info)
            else:
                new_states = env.step(states, actions, dones, info)
            sink_states_mask = new_states.is_sink_state

            step += 1

            new_dones = (
                new_states.is_initial_state
                if self.estimator.is_backward
                else sink_states_mask
            ) & ~dones
            # all values from
            if new_dones.any():
                print("new done state!")
            trajectories_dones[new_dones & ~dones] = step
            # NOTE: step 3 -- get log reward

            if self.estimator.is_backward:
                trajectories_log_rewards[new_dones & ~dones], res = env.log_reward(
                    backward_state[new_dones & ~dones], info
                )
            else:
                # NOTE: current is new state
                trajectories_log_rewards[new_dones & ~dones], res = env.log_reward(
                    states[new_dones & ~dones], info
                )


            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = env.States(tensor=trajectories_states)
        trajectories_actions = env.Actions.stack(trajectories_actions)
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0)

        trajectories = Trajectories(
            env=env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.estimator.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
            res=res,
            start_state=start_state,
        )

        return trajectories
