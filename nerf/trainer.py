import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import imageio
from tqdm import tqdm
import os

class ObjNeRFTrainer:
    def __init__(self,model,optimizer,cfg,device,out_dir):
        self.model=model
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir=out_dir

    def train_step(self,data):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self,data,pytest=False,raw_noise_std=0,train=True,white_bkgd=True):
        device=self.device
        loss=0.

        target_pixels=data.get('target_pixels').to(device)  # (B,3)
        input_origins=data.get('input_origins').to(device)  # (B,3)
        input_rays=data.get('input_rays').to(device)        # (B,3)
        B,_=target_pixels.shape
        near=2.
        far=6.
        input_dirs = input_rays / torch.norm(input_rays, dim=-1, keepdim=True)  # (B,3)
        pts, z_vals=self.coarse_sampling(input_origins,input_rays,near,far,self.config["N_coarse_samples"],train,pytest)

        rgb, alpha=self.model.coarse_nerf(pts,input_dirs)  # (B,N_coarse_samples,3) (B,N_coarse_samples,1)
        coarse_rgb_map, weights=self.volume_rendering(rgb,alpha,input_rays,z_vals,white_bkgd,raw_noise_std,pytest)
        # z sampling
        z_vals=0.5 * (z_vals[:,1:] + z_vals[:,:-1])
        pts,z_vals = self.fine_sampling(input_origins,input_rays,z_vals, weights[:,1:-1,0], self.config['N_fine_samples'], pytest,train) # (B, N_fine_samples)
        pts,z_vals = pts.detach(), z_vals.detach()
        rgb, alpha=self.model.fine_nerf(pts,input_dirs)  # (B,N_fine_samples,3) (B,N_fine_samples,1)

        # raw2outputs
        fine_rgb_map,_=self.volume_rendering(rgb,alpha,input_rays,z_vals,white_bkgd,raw_noise_std,pytest)

        coarse_mse_loss=torch.mean((coarse_rgb_map - target_pixels) ** 2)
        fine_mse_loss=torch.mean((fine_rgb_map - target_pixels) ** 2)
        loss+=coarse_mse_loss+fine_mse_loss
        return loss

    def volume_rendering(self,rgb,alpha,input_rays,z_vals,white_bkgd,raw_noise_std,pytest):
        """
        volume rendring用の函数です。
        Parameters
        ----------
        rgb: サンプリング点ごとのRGB size=(B,N,3)
        alpha: サンプリング点ごとのalpha size=(B,N,1)
        input_rays: レイのベクトル size=(B,3)
        z_vals: サンプリング点のレイにおける相対位置 (from 0 to 1) size=(B,N)
        """
        device=self.device
        dists = z_vals[:,1:] - z_vals[:,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[:,:1].shape).to(device)], -1) # 一個少ないので一番奥に無限距離を追加
        dists = dists * torch.norm(input_rays.unsqueeze(1), dim=-1) # レイの長さをかけることでサンプリング点間の距離にする

        rgb = torch.sigmoid(rgb)  # (B, N, 3)
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(alpha.shape) * raw_noise_std
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(alpha.shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha=1.-torch.exp(-F.relu(alpha+noise)*dists.unsqueeze(2))  # (B, N, 1)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0],1, 1)).to(device), 1.-alpha + 1e-10], 1), 1)[:, :-1] # (B, N, 1)
        rgb_map = torch.sum(weights * rgb, -2)  # (B, 3)
        acc_map = torch.sum(weights, 1)
        if white_bkgd:
            rgb_map = rgb_map + (1.0-acc_map)
        return rgb_map, weights

    def coarse_sampling(self, input_origins,input_rays,near,far,N_samples,train,pytest):
        device=self.device
        B,D=input_origins.shape
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # (N_coarse_samples, )
        z_vals = z_vals.unsqueeze(0).repeat(B,1).to(device)  # (B, N_coarse_samples)
        # render_rays
        if train:
            # 訓練時のみゆらぎを持たせる。
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(device)
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand).to(device)
            z_vals = lower + (upper - lower) * t_rand
        pts = input_origins.unsqueeze(1) + input_rays.unsqueeze(1) * z_vals.unsqueeze(2) # (B, N_coarse_samples, 3)
        return pts, z_vals

    def fine_sampling(self,input_origins,input_rays,z_vals,weights,N_samples,pytest,train):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        if train:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
            if train:
                u = np.random.rand(*new_shape)
            u = torch.from_numpy(u.astype(np.float32)).clone()

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u.to(self.device), right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        z_vals_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u.to(self.device)-cdf_g[...,0])/denom.to(self.device)
        z_samples = z_vals_g[...,0] + t * (z_vals_g[...,1]-z_vals_g[...,0])
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = input_origins.unsqueeze(1) + input_rays.unsqueeze(1) * z_vals.unsqueeze(2) # (B, N_fine_samples, 3)
        return pts, z_vals

    def visualize(self,demo_origins,demo_rays,chunk=512,white_bkgd=True):
        self.model.eval()
        device=self.device
        B,H,W,D=demo_origins.shape
        demo_origins=demo_origins.reshape(B*H*W,3).to(device)
        demo_rays=demo_rays.reshape(B*H*W,3).to(device)
        fine_rgb_maps=[]
        for C in tqdm(range(0, B*H*W, chunk)):
            input_origins=demo_origins[C:C+chunk]
            input_rays=demo_rays[C:C+chunk]
            fine_rgb_map=self.render(input_origins,input_rays)
            fine_rgb_maps.append((255*np.clip(fine_rgb_map.detach().cpu().numpy(),0,1)).astype(np.uint8))
        rgbs=np.concatenate(fine_rgb_maps,axis=0).reshape(B,H,W,3)
        os.makedirs(os.path.join(self.out_dir,'demo'),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,'demo/images'),exist_ok=True)
        for i in range(rgbs.shape[0]):
            filename = os.path.join(self.out_dir,'demo/images/', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgbs[i])
        imageio.mimwrite(os.path.join(self.out_dir,'demo/', 'video.mp4'), rgbs, fps=30, quality=8)
        return

    def render(self,origins,rays,white_bkgd=True):
        device=self.device
        origins=origins.to(device)
        rays=rays.to(device)
        near=2.
        far=6.
        dirs = rays / torch.norm(rays, dim=-1, keepdim=True)  # (B,3)
        pts,z_vals=self.coarse_sampling(origins,rays,near,far,self.config['N_coarse_samples'],False,True)
        rgb, alpha=self.model.coarse_nerf(pts,dirs)  # (B,N_coarse_samples,3) (B,N_coarse_samples,1)
        # raw2outputs
        _,weights=self.volume_rendering(rgb,alpha,rays,z_vals,white_bkgd,False,True)
        z_vals=0.5 * (z_vals[:,1:] + z_vals[:,:-1])
        pts,z_vals = self.fine_sampling(origins, rays, z_vals, weights[:,1:-1,0], self.config['N_fine_samples'], True, False) # (B, N_fine_samples)
        pts,z_vals = pts.detach(), z_vals.detach()
        rgb, alpha=self.model.fine_nerf(pts,dirs)  # (B,N_fine_samples,3) (B,N_fine_samples,1)
        # raw2outputs
        fine_rgb_map,_=self.volume_rendering(rgb,alpha,rays,z_vals,white_bkgd,False,True)
        return fine_rgb_map
