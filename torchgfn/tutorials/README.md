# Welcome to the `torchgfn` tutorials (vanilla introduction)

1. Learn the building blocks of your GflowNet with [notebooks](https://github.com/saleml/torchgfn/tree/master/tutorials/notebooks/)
2. See `torchgfn` in action with example [training scripts](https://github.com/saleml/torchgfn/tree/master/tutorials/examples/)
3. Read a summary of what you need to do to create your own [Environment](https://github.com/saleml/torchgfn/tree/master/tutorials/ENV.md)

# How to install it?

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

# About the implementation the Meta Schedule environment in MLC

- The the implementation of Env `MetaScheduleEnv` can be found in `torchgfn/src/gfn/gym/mlc_meta_schedule.py`
- The the implementation of Embedding Strategies `` can be found in `torchgfn/mlc_dataset/mlc_dataset.py` 

