import random
from datetime import datetime

import numpy
import torch


def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(category, message):
    timestamp = datetime.now().strftime("%m/%d - %H:%M")
    print(f"[{category.upper()}] [{timestamp}] {message}")
