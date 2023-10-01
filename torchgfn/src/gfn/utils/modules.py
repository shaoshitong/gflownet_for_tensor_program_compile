"""This file contains some examples of modules that can be used with GFN."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType as TT

from torchvision import datasets, transforms
from torchvision import models


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 1, 28, 28)
        output = self.conv1(x)
        # print("out1 shape = ", output.shape)
        output = self.conv2_x(output)
        # print("out2 shape = ", output.shape)
        output = self.conv3_x(output)
        # print("out3 shape = ", output.shape)
        output = self.conv4_x(output)
        # print("out4 shape = ", output.shape)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = output.view(bs, -1)
        print("final shape = ", output.shape)
        return output

def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + (([nn.BatchNorm1d(o), act] if with_bn else [act]) if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []
    ) + tail))
    return net

class NeuralNet(nn.Module):
    """Implements a basic MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 3,
        activation_fn = "relu",
        torso: Optional[nn.Module] = None,
    ):
        """Instantiates a MLP instance.

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
            hidden_dim: Number of units per hidden layer.
            n_hidden_layers: Number of hidden layers.
            activation_fn: Activation function.
            torso: If provided, this module will be used as the torso of the network
                (i.e. all layers except last layer).
        """
        super().__init__()
        self._output_dim = output_dim

        if torso is None:
            assert (
                n_hidden_layers is not None and n_hidden_layers >= 0
            ), "n_hidden_layers must be >= 0"
            assert activation_fn is not None, "activation_fn must be provided"

            if activation_fn == "elu":
                activation = nn.ELU
            elif activation_fn == "relu":
                activation = nn.ReLU
            elif activation_fn == "tanh":
                activation = nn.Tanh

            # self.torso = [nn.Linear(input_dim, hidden_dim), activation()]
            # for _ in range(n_hidden_layers - 1):
            #     self.torso.append(nn.Linear(hidden_dim, hidden_dim))
            #     self.torso.append(activation())
            # self.torso = nn.Sequential(*self.torso)
            # input_dim --> hid dim --> 3*input_dim
            self.torso = make_mlp([input_dim] + [hidden_dim] * n_hidden_layers +
                              [3 * input_dim], act=nn.LeakyReLU(), with_bn=False)
            # self.torso = ResNet(BottleNeck, [3, 4, 6, 3], input_dim)
            # self.torso.hidden_dim = hidden_dim
        else:
            self.torso = torso
        self.last_layer = nn.Linear(3*input_dim, output_dim)
        self.device = None

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        """Forward method for the neural network.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the MLP.
        Returns: out, a set of continuous variables.
        """
        if self.device is None:
            self.device = preprocessed_states.device
            self.to(self.device)
        out = self.torso(preprocessed_states)
        out = self.last_layer(out)
        return out


class Tabular(nn.Module):
    """Implements a tabular policy.

    This class is only compatible with the EnumPreprocessor.

    Attributes:
        table: a tensor with dimensions [n_states, output_dim].
        device: the device that holds this policy.
    """

    def __init__(self, n_states: int, output_dim: int) -> None:
        """
        Args:
            n_states (int): Number of states in the environment.
            output_dim (int): Output dimension.
        """

        super().__init__()

        self.table = torch.zeros(
            (n_states, output_dim),
            dtype=torch.float,
        )

        self.table = nn.parameter.Parameter(self.table)
        self.device = None

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        if self.device is None:
            self.device = preprocessed_states.device
            self.table = self.table.to(self.device)
        assert preprocessed_states.dtype == torch.long
        outputs = self.table[preprocessed_states.squeeze(-1)]
        return outputs


class DiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out
