"""Dump relay IR and task information for networks"""

import argparse
import os
from typing import List, Tuple

from tqdm import tqdm  # type: ignore
from tvm.meta_schedule.testing.relay_workload import get_network


# pylint: disable=too-many-branches
def _build_dataset():
    network_keys = []

    # bert
    for batch_size in [1, 4, 8]:
        for seq_length in [64, 128, 256]:
            for scale in ['tiny', 'base', 'medium', 'large']:
                network_keys.append((f'bert_{scale}',
                                    [batch_size, seq_length]))

    # dcgan
    for batch_size in [1, 4, 8]:
        for image_size in [64, 80, 96]:
            network_keys.append((f'dcgan',
                                [batch_size, 3, image_size, image_size]))


    # resnet_18 and resnet_50
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                network_keys.append((f'resnet_{layer}',
                                    [batch_size, 3, image_size, image_size]))

    # mobilenet_v2
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2', 'mobilenet_v3']:
                network_keys.append((f'{name}',
                                    [batch_size, 3, image_size, image_size]))

    # wide-resnet
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}',
                                    [batch_size, 3, image_size, image_size]))

    # resnext
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnext_{layer}',
                                    [batch_size, 3, image_size, image_size]))

    # inception-v3
    for batch_size in [1, 2, 4]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                [batch_size, 3, image_size, image_size]))

    # densenet
    for batch_size in [1, 2, 4]:
        for image_size in [224, 240, 256]:
            network_keys.append((f'densenet_121',
                                [batch_size, 3, image_size, image_size]))

    # resnet3d
    for batch_size in [1, 2, 4]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}',
                                    [batch_size, 3, image_size, image_size, 16]))
                
    return network_keys

def main():
    model_cache_dir = args.model_cache_dir
    try:
        os.makedirs(model_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {model_cache_dir} cannot be created successfully.")
    keys = _build_dataset()
    for name, input_shape in tqdm(keys):
        get_network(name=name, input_shape=input_shape, cache_dir=model_cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="Please provide the full path to the model cache dir.",
    )
    args = parser.parse_args()  # pylint: disable=invalid-name
    main()