import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from nerf.data.blender import load_blender,focus_sample,defocus_sample
from nerf.utils.common import init_ddp
from nerf.checkpoint import Checkpoint
from nerf.lr_scheduler import LrScheduler
from nerf.trainer import NeRFTrainer
from nerf.model import NeRFModel

if __name__ == '__main__':
    config='runs/lego/config.yaml'
    out_dir = os.path.dirname(config)
    with open(config) as f:
        cfg=yaml.load(f, Loader=yaml.CLoader)
    rank, world_size = init_ddp()
    print('rank = ',rank,' world size = ',world_size)
    device = torch.device(f"cuda:{rank}")
    batch_size = cfg['training']['batch_size'] // world_size
    ray_size=cfg['training']['ray_size']
    # load data
    blender_data=load_blender(os.path.join('data/',cfg['data']['dataset']),'train')
    target_images=torch.from_numpy(blender_data['target_images'].astype(np.float32)).clone()
    input_origins=torch.from_numpy(blender_data['input_origins'].astype(np.float32)).clone()
    input_rays=torch.from_numpy(blender_data['input_rays'].astype(np.float32)).clone()
    # lr
    peak_lr=cfg['training']['peak_lr']
    peak_it=cfg['training']['peak_it']
    decay_rate=cfg['training']['decay_rate']
    decay_it=cfg['training']['decay_it']
    lr_scheduler = LrScheduler(peak_lr=peak_lr, peak_it=peak_it, decay_rate=decay_rate, decay_it=decay_it)
    model=NeRFModel(cfg["model"]).to(device)
    if world_size > 1:
        model.fine_nerf = DistributedDataParallel(model.fine_nerf, device_ids=[rank], output_device=rank)
        model.coarse_nerf = DistributedDataParallel(model.coarse_nerf, device_ids=[rank], output_device=rank)
        fine_nerf_module = model.fine_nerf.module
        coarse_nerf_module = model.coarse_nerf.module
    else:
        fine_nerf_module = model.fine_nerf
        coarse_nerf_module = model.coarse_nerf
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0),betas=(0.9, 0.999))
    checkpoint = Checkpoint(out_dir, device=device, fine_nerf=fine_nerf_module,
                            coarse_nerf=coarse_nerf_module, optimizer=optimizer)
    try:
        load_dict = checkpoint.load('model.pt')
        print('Model loaded')
    except FileNotFoundError:
        load_dict = dict()
    except RuntimeError:
        load_dict = dict()
        print("Faild loading")
    trainer=NeRFTrainer(model,optimizer,cfg,device,out_dir)
    # train
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    losses=[]
    while True:
        epoch_it+=1
        for _ in range(0,len(target_images),batch_size):
            it += 1
            if rank == 0:
                checkpoint_scalars = {'epoch_it': epoch_it, 'it': it}
                # Save checkpoint
                if it % 2000 == 0:
                    checkpoint.save('model.pt', **checkpoint_scalars)
            new_lr = lr_scheduler.get_cur_lr(it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            # start
            img_i = np.random.choice(len(target_images))
            target_image=target_images[img_i]
            input_ray=input_rays[img_i]
            input_origin=input_origins[img_i]
            H,W,D=input_ray.shape
            if it < cfg['training']['precrop_iters']:
                iH,iW=focus_sample(H,W,cfg['training']['precrop_frac'],ray_size)
            else:
                iH,iW=defocus_sample(H,W,ray_size)
            input_origin = input_origin[iH, iW]  # (N_rand, 3)
            input_ray = input_ray[iH, iW]  # (N_rand, 3)
            target_pixels = target_image[iH, iW]  # (N_rand, 3)
            batch={'target_pixels':target_pixels,'input_origins':input_origin,'input_rays':input_ray}
            loss=trainer.train_step(batch)
            losses.append(loss)
            if it % 100==0:
                print('epoch... ',epoch_it,' it... ',it,' loss... ',sum(losses)/len(losses),' lr...', new_lr)
                losses=[]