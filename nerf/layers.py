import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=15, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape
        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        while len(multipliers.shape) < len(coords.unsqueeze(-1).shape):
            multipliers = multipliers.unsqueeze(0)
        scaled_coords = coords.unsqueeze(-1) * multipliers
        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((coords, sines, cosines), -1)
        return result

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, num_pts_octaves=15, start_pts_octave=0,num_view_octaves=15, start_view_octave=0, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        input_ch = (num_pts_octaves*2+1)*3
        input_ch_views = (num_view_octaves*2+1)*3
        self.skips = skips
        self.pts_pe=PositionalEncoding(num_pts_octaves,start_pts_octave)
        self.view_pe=PositionalEncoding(num_view_octaves,start_view_octave)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, pts, view_dirs):
        B,N,D=pts.shape
        B,D=view_dirs.shape
        view_dirs=view_dirs.unsqueeze(1).expand(pts.shape)
        pts=self.pts_pe(pts)
        view_dirs=self.view_pe(view_dirs)
        h = pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, view_dirs], -1)
    
        for i, _ in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        return rgb, alpha