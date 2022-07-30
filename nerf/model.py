import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from nerf.layers import NeRF

class NeRFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse_nerf=NeRF(**cfg['nerf_kwargs'])
        self.fine_nerf=NeRF(**cfg['nerf_kwargs'])