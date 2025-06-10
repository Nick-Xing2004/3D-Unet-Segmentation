import torch
import random
import argparse
import numpy as np

#unify seed values for all random number generators to ensure reproducibility
def customize_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Uncomment below if using multi-GPU setup
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)