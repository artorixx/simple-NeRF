import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from nerf.data.blender import load_demo
from nerf.utils.common import init_ddp
from nerf.checkpoint import Checkpoint
from nerf.lr_scheduler import LrScheduler
from nerf.trainer import ObjNeRFTrainer
from nerf.model import ObjNeRF

if __name__ == '__main__':
    config='runs/lego/config.yaml'
    out_dir = os.path.dirname(config)
    with open(config) as f:
        cfg=yaml.load(f, Loader=yaml.CLoader)
    rank, world_size = init_ddp()
    print('rank = ',rank,' world size = ',world_size)
    device = torch.device(f"cuda:{rank}")
    batch_size = 1 # cfg['training']['batch_size'] // world_size
    # load data
    demo_data=load_demo(os.path.join('data/',cfg['data']['dataset']),'train')
    demo_origins=torch.from_numpy(demo_data['demo_origins'].astype(np.float32)).clone()
    demo_rays=torch.from_numpy(demo_data['demo_rays'].astype(np.float32)).clone()
    model=ObjNeRF(cfg["model"]).to(device)
    if world_size > 1:
        model.fine_nerf = DistributedDataParallel(model.fine_nerf, device_ids=[rank], output_device=rank)
        model.coarse_nerf = DistributedDataParallel(model.coarse_nerf, device_ids=[rank], output_device=rank)
        fine_nerf_module = model.fine_nerf.module
        coarse_nerf_module = model.coarse_nerf.module
    else:
        fine_nerf_module = model.fine_nerf
        coarse_nerf_module = model.coarse_nerf
    lr_scheduler = LrScheduler()
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0),betas=(0.9, 0.999))
    checkpoint = Checkpoint(out_dir, device=device, fine_nerf=fine_nerf_module,
                            coarse_nerf=coarse_nerf_module, optimizer=optimizer)
    checkpoint.load('model.pt')
    trainer=ObjNeRFTrainer(model,optimizer,cfg,device,out_dir)
    trainer.visualize(demo_origins,demo_rays,chunk=1024)

