# torchgfn: a Python package for GFlowNets

<p align="center"> Please cite <a href="https://arxiv.org/abs/2305.14594">this paper</a> if you are using the library for your research </p>

## Installing the package

To install the cutting edge version (from the `main` branch):

```bash
git clone https://github.com/shaoshitong/gflownet_for_tensor_program_compile
cd torchgfn
pip install .
```

## About this repo

This repo serves the purpose of fast prototyping [GFlowNet](https://arxiv.org/abs/2111.09266) (GFN) related algorithms. It decouples the environment definition, the sampling process, and the parametrization of the function approximators used to calculate the GFN loss. It aims to accompany researchers and engineers in learning about GFlowNets, and in developing new algorithms.

Currently, the library is shipped with three environments: two discrete environments (Discrete Energy Based Model and Hyper Grid) and a continuous box environment. The library is designed to allow users to define their own environments. See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md) for more details.

### Scripts and notebooks

Example scripts and notebooks for the three environments are provided [here](https://github.com/saleml/torchgfn/tree/master/tutorials/examples). For the hyper grid and the box environments, the provided scripts are supposed to reproduce published results.

## Details about the codebase

### Defining an environment

See [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md)

### States

States are the primitive building blocks for GFlowNet objects such as transitions and trajectories, on which losses operate.

An abstract `States` class is provided. But for each environment, a `States` subclass is needed. A `States` object
is a collection of multiple states (nodes of the DAG). A tensor representation of the states is required for batching. If a state is represented with a tensor of shape `(*state_shape)`, a batch of states is represented with a `States` object, with the attribute `tensor` of shape `(*batch_shape, *state_shape)`. Other
representations are possible (e.g. a state as a string, a `numpy` array, a graph, etc...), but these representations cannot be batched, unless the user specifies a function that transforms these raw states to tensors.

The `batch_shape` attribute is required to keep track of the batch dimension. A trajectory can be represented by a States object with `batch_shape = (n_states,)`. Multiple trajectories can be represented by a States object with `batch_shape = (n_states, n_trajectories)`.

Because multiple trajectories can have different lengths, batching requires appending a dummy tensor to trajectories that are shorter than the longest trajectory. The dummy state is the $s_f$ attribute of the environment (e.g. `[-1, ..., -1]`, or `[-inf, ..., -inf]`, etc...). Which is never processed, and is used to pad the batch of states only.

For discrete environments, the action set is represented with the set $\{0, \dots, n_{actions} - 1\}$, where the $(n_{actions})$-th action always corresponds to the exit or terminate action, i.e. that results in a transition of the type $s \rightarrow s_f$, but not all actions are possible at all states. For discrete environments, each `States` object is endowed with two extra attributes: `forward_masks` and `backward_masks`, representing which actions are allowed at each state and which actions could have led to each state, respectively. Such states are instances of the `DiscreteStates` abstract subclass of `States`. The `forward_masks` tensor is of shape `(*batch_shape, n_{actions})`, and `backward_masks` is of shape `(*batch_shape, n_{actions} - 1)`. Each subclass of `DiscreteStates` needs to implement the `update_masks` function, that uses the environment's logic to define the two tensors.

### Actions
Actions should be though of as internal actions of an agent building a compositional object. They correspond to transitions $s \rightarrow s'$. An abstract `Actions` class is provided. It is automatically subclassed for discrete environments, but needs to be manually subclassed otherwise.

Similar to `States` objects, each action is a tensor of shape `(*batch_shape, *action_shape)`. For discrete environments for instances, `action_shape = (1,)`, representing an integer between $0$ and $n_{actions} - 1$.

Additionally, each subclass needs to define two more class variable tensors:
- `dummy_action`: A tensor that is padded to sequences of actions in the shorter trajectories of a batch of trajectories. It is `[-1]` for discrete environments.
- `exit_action`: A tensor that corresponds to the termination action. It is `[n_{actions} - 1]` fo discrete environments.

### Containers

Containers are collections of `States`, along with other information, such as reward values, or densities $p(s' \mid s)$. Two containers are available:

- [Transitions](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/transitions.py), representing a batch of transitions $s \rightarrow s'$.
- [Trajectories](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/trajectories.py), representing a batch of complete trajectories $\tau = s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n \rightarrow s_f$.

These containers can either be instantiated using a `States` object, or can be initialized as empty containers that can be populated on the fly, allowing the usage of the [ReplayBuffer](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/replay_buffer.py) class.

They inherit from the base `Container` [class](https://github.com/saleml/torchgfn/tree/master/src/gfn/containers/base.py), indicating some helpful methods.

In most cases, one needs to sample complete trajectories. From a batch of trajectories, a batch of states and batch of transitions can be defined using `Trajectories.to_transitions()` and `Trajectories.to_states()`, in order to train GFlowNets with losses that are edge-decomposable or state-decomposable.  These exclude meaningless transitions and dummy states that were added to the batch of trajectories to allow for efficient batching.

### Modules

Training GFlowNets requires one or multiple estimators, called `GFNModule`s, which is an abstract subclass of `torch.nn.Module`. In addition to the usual `forward` function, `GFNModule`s need to implement a `required_output_dim` attribute, to ensure that the outputs have the required dimension for the task at hand; and some (but not all) need to implement a `to_probability_distribution` function.

- `DiscretePolicyEstimator` is a `GFNModule` that defines the policies $P_F(. \mid s)$ and $P_B(. \mid s)$ for discrete environments. When `is_backward=False`, the required output dimension is `n = env.n_actions`, and when `is_backward=True`, it is `n = env.n_actions - 1`. These `n` numbers represent the logits of a Categorical distribution. The corresponding `to_probability_distribution` function transforms the logits by masking illegal actions (according to the forward or backward masks), then return a Categorical distribution. The masking is done by setting the corresponding logit to $-\infty$. The function also includes exploration parameters, in order to define a tempered version of $P_F$, or a mixture of $P_F$ with a uniform distribution. `DiscretePolicyEstimator`` with `is_backward=False`` can be used to represent log-edge-flow estimators $\log F(s \rightarrow s')$.
- `ScalarModule` is a simple module with required output dimension 1. It is useful to define log-state flows $\log F(s)$.

For non-discrete environments, the user needs to specify their own policies $P_F$ and $P_B$. The module, taking as input a batch of states (as a `States`) object, should return the batched parameters of a `torch.Distribution`. The distribution depends on the environment. The `to_probability_distribution` function handles the conversion of the parameter outputs to an actual batched `Distribution` object, that implements at least the `sample` and `log_prob` functions. An example is provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/gym/helpers/box_utils.py), for a square environment in which the forward policy has support either on a quarter disk, or on an arc-circle, such that the angle, and the radius (for the quarter disk part) are scaled samples from a mixture of Beta distributions. The provided example shows an intricate scenario, and it is not expected that user defined environment need this much level of details.

In all `GFNModule`s, note that the input of the `forward` function is a `States` object. Meaning that they first need to be transformed to tensors. However, `states.tensor` does not necessarily include the structure that a neural network can used to generalize. It is common in these scenarios to have a function that transforms these raw tensor states to ones where the structure is clearer, via a `Preprocessor` object, that is part of the environment. More on this [here](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md). The default preprocessor of an environment is the identity preprocessor. The `forward` pass thus first calls the `preprocessor` attribute of the environment on `States`, before performing any transformation. The `preprocessor` is thus an attribute of the module. If it is not explicitly defined, it is set to the identity preprocessor.

For discrete environments, a `Tabular` module is provided, where a lookup table is used instead of a neural network. Additionally, a `UniformPB` module is provided, implementing a uniform backward policy. These modules are provided [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/utils/modules.py).

### Samplers

A [Sampler](https://github.com/saleml/torchgfn/tree/master/src/gfn/samplers.py) object defines how actions are sampled (`sample_actions()`) at each state, and trajectories  (`sample_trajectories()`), which can sample a batch of trajectories starting from a given set of initial states or starting from $s_0$. It requires a `GFNModule` that implements the `to_probability_distribution` function. For off-policy sampling, the parameters of `to_probability_distribution` can be directly passed when initializing the `Sampler`.


### Losses

GFlowNets can be trained with different losses, each of which requires a different parametrization, which we call in this library a `GFlowNet`. A `GFlowNet` is a `GFNModule` that includes one or multiple `GFNModule`s, at least one of which implements a `to_probability_distribution` function. They also need to implement a `loss` function, that takes as input either states, transitions, or trajectories, depending on the loss.

Currently, the implemented losses are:

- Flow Matching
- Detailed Balance (and it's modified variant).
- Trajectory Balance
- Sub-Trajectory Balance. By default, each sub-trajectory is weighted geometrically (within the trajectory) depending on its length. This corresponds to the strategy defined [here](https://www.semanticscholar.org/reader/f2c32fe3f7f3e2e9d36d833e32ec55fc93f900f5). Other strategies exist and are implemented [here](https://github.com/saleml/torchgfn/tree/master/src/gfn/losses/sub_trajectory_balance.py).
- Log Partition Variance loss. Introduced [here](https://arxiv.org/abs/2302.05446)

### Extending GFlowNets

To define a new `GFlowNet`, the user needs to define a class which subclasses `GFlowNet` and implements the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

Based on the type of training samples returned by `to_training_samples`, the user should define the generic type `TrainingSampleType` when subclassing `GFlowNet`. For example, if the training sample is an instance of `Trajectories`, the `GFlowNet` class should be subclassed as `GFlowNet[Trajectories]`. Thus, the class definition should look like this:

```python
class MyGFlowNet(GFlowNet[Trajectories]):
    ...
```

**Example: Flow Matching GFlowNet**

Let's consider the example of the `FMGFlowNet` class, which is a subclass of `GFlowNet` that implements the Flow Matching GFlowNet. The training samples are tuples of discrete states, so the class references the type `Tuple[DiscreteStates, DiscreteStates]` when subclassing `GFlowNet`:

```python
class FMGFlowNet(GFlowNet[Tuple[DiscreteStates, DiscreteStates]]):
    ...

    def to_training_samples(
        self, trajectories: Trajectories
    ) -> tuple[DiscreteStates, DiscreteStates]:
        """Converts a batch of trajectories into a batch of training samples."""
        return trajectories.to_non_initial_intermediary_and_terminating_states()

```

**Adding New Training Sample Types**

If your GFlowNet returns a unique type of training samples, you'll need to expand the `TrainingSampleType` bound. This ensures type-safety and better code clarity.

In the earlier example, the `FMGFlowNet` used:

```python
GFlowNet[Tuple[DiscreteStates, DiscreteStates]]
```

This means the method `to_training_samples` should return a tuple of `DiscreteStates`.

If the `to_training_sample` method of your new GFlowNet, for example, returns an `int`, you should expand the `TrainingSampleType` in `src/gfn/gflownet/base.py` to include this type in the `bound` of the `TypeVar`:

Before:

```python
TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, tuple[States, ...]]
)
```

After:

```python
TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, tuple[States, ...], int]
)
```

**Implementing Class Methods**

As mentioned earlier, your new GFlowNet must implement the following methods:

- `sample_trajectories`: Sample a specific number of complete trajectories.
- `loss`: Compute the loss given the training objects.
- `to_training_samples`: Convert trajectories to training samples.

These methods are defined in `src/gfn/gflownet/base.py` and are abstract methods, so they must be implemented in your new GFlowNet. If your GFlowNet has unique functionality which should be represented as additional class methods, implement them as required. Remember to document new methods to ensure other developers understand their purposes and use-cases!

**Testing**

Remember to create unit tests for your new GFlowNet to ensure it works as intended and integrates seamlessly with other parts of the codebase. This ensures maintainability and reliability of the code!



# Update by Shitong Shao

## How to install it?

The installation of `torchgfn` is vert simple, you can run:

```bash
cd [/path/to/torchgfn]
pip install .
```

Please note that every modification you make will require a fresh install to call it properly! For instance:

```bash
# you make some modifications
cd [/path/to/torchgfn]
pip uninstall torchgfn && pip install .
```

## About the implementation the Meta Schedule environment in MLC

- The implementation of Env `MetaScheduleEnv` can be found in `torchgfn/src/gfn/gym/mlc_meta_schedule.py`
- The implementation of Embedding Strategies `GflowNetEmbedding` can be found in `torchgfn/mlc_dataset/mlc_dataset.py`
- The implementation of Load GflowNet Dataset and Produce GflowNet Dataset can be found in `torchgfn/mlc_dataset/mlc_dataset.py`, where `gflownet_data_save` is used to make a gflownet dataset, `MLCGflowNetDataset` is the formal GflowNet Dataset (inheritance to `torch.utils.data.Dataset`), and `gflownet_data_load` is used to load a gflownet dataloader (i.e. `torch.utils.data.DataLoader`).
- The implementation of Scripts can be found in `torchgfn/src/gfn/gym/save_gflownet_dataset.py`, `torchgfn/src/gfn/gym/train_discrete_edm.py`, and `torchgfn/src/gfn/gym/train_meta_schedule.py`. As they are named, where `torchgfn/src/gfn/gym/save_gflownet_dataset.py` is used to produce the GflowNet dataset, which has been produced so far. `torchgfn/src/gfn/gym/train_discrete_edm.py` is used for the validation of the environment for discrete edm. `torchgfn/src/gfn/gym/train_meta_schedule.py` is used for the validation of the environment for meta schedule. Currently, both `torchgfn/src/gfn/gym/train_discrete_edm.py` and `torchgfn/src/gfn/gym/train_meta_schedule.py` still have problems, and it is necessary to ensure that B can generate the discrete MNIST dataset correctly before we can ensure that the training of `torchgfn/src/gfn/gym/train_meta_schedule.py` is correctly.
- The implementation of Backward Training Paradigm can be found in `torchgfn/src/gfn/gflownet/base.py` line 109:168. Since I implemented this, the current backsampling-based trajectory training may be incorrect, but this has to be implemented in the future.

