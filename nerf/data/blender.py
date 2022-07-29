import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from nerf.utils.nerf import get_rays_np


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender(path,mode,half_res=True,white_bkgd=True):
    with open(os.path.join(path, 'transforms_{}.json'.format(mode)), 'r') as f:
        metadata = json.load(f)
    imgs=[]
    poses=[]
    for frame in metadata['frames']:
        file_path = os.path.join(path, frame['file_path'] + '.png')
        imgs.append(imageio.imread(file_path))
        poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    N, H, W, D = imgs.shape
    camera_angle_x = float(metadata['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res # (N,H,W,C)
    if white_bkgd:
        imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
    else:
        imgs = imgs[...,:3]
    target_images=imgs.reshape(N,H,W,3)
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    near=2.
    far=6.
    input_origins=[]
    input_rays=[]
    for p in poses[:,:3,:4]:
        input_origin,input_ray=get_rays_np(H, W, K, p)
        input_origins.append(input_origin)
        input_rays.append(input_ray)
    input_origins=np.stack(input_origins,axis=0).reshape(N,H,W,3) # (N*H*W, 3)
    input_rays=np.stack(input_rays,axis=0).reshape(N,H,W,3) # (N*H*W, 3)
    imsize=(H,W)
    result={
        'target_images': target_images,
        'input_origins': input_origins,
        'input_rays': input_rays,
        'near': near,
        'far': far,
        'imsize': imsize,
    }
    return result

def load_demo(path,mode,half_res=True):
    with open(os.path.join(path, 'transforms_{}.json'.format(mode)), 'r') as f:
        metadata = json.load(f)
    file_path = os.path.join(path, metadata['frames'][0]['file_path'] + '.png')
    img=imageio.imread(file_path)
    img = (np.array(img) / 255.).astype(np.float32)
    H, W, _ = img.shape
    camera_angle_x = float(metadata['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    # for demo
    demo_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    demo_origins=[]
    demo_rays=[]
    for p in demo_poses[:,:3,:4]:
        demo_origin,demo_ray=get_rays_np(H, W, K, p)
        demo_origins.append(demo_origin)
        demo_rays.append(demo_ray)
    demo_origins=np.stack(demo_origins,axis=0)
    demo_rays=np.stack(demo_rays,axis=0)
    result={
        'demo_origins': demo_origins,
        'demo_rays': demo_rays
    }
    return result

def focus_sample(H,W,frac,ray_size):
    dH = int(H//2 * frac)
    dW = int(W//2 * frac)
    N=4*dH*dW
    random_ids=np.random.choice(N,size=ray_size,replace=False)
    iH=random_ids//(2*dW)
    iW=random_ids%(2*dW)
    iH=H//2-dH+iH
    iW=W//2-dW+iW
    iH,iW=list(iH),list(iW)
    return iH,iW

def defocus_sample(H,W,ray_size):
    random_ids=np.random.choice(H*W,size=ray_size,replace=False)
    iH=random_ids//(W)
    iW=random_ids%(W)
    iH,iW=list(iH),list(iW)
    return iH,iW