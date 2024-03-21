# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random

import numpy as np
import torch

# from llama_recipes.utils.memory_utils import MemoryTrace
# from llama_recipes.utils.dataset_utils import *
# from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
# from llama_recipes.utils.train_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    