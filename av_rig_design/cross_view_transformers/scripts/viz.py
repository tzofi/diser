import os
from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra
import numpy as np

import torch
import time
import imageio
import ipywidgets as widgets

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback
from cross_view_transformer.metrics import IoUMetric

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)
    print("Got data module")
    loader = data_module.train_dataloader()
    print("Got loader")
    CHECKPOINT_PATH = os.path.join(cfg.experiment.checkpoint_path,"model-v1.ckpt")
    network = load_backbone(CHECKPOINT_PATH)
    print("Checkpoint loaded!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()

    images = list()
    metric = IoUMetric(label_indices=None)
    print("Starting evaluation!")

    H=1084
    W=1924
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    counter = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            #print("Processing batch {}!".format(i))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch)

            imgs = batch['image'].detach().cpu().numpy()
            binimgs = batch['bev'].detach().cpu().numpy()
            metric.update(pred,batch)

            pred = pred['bev']
            out = pred.sigmoid().cpu()
            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = np.moveaxis(img,0,2) #* 255.0 
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Ground truth')
                ], loc=(0.01, 0.86))
                #print(out.shape, binimgs.shape)
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Oranges', alpha=0.3)

                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = f'/var/job/998e58/scratch/zfs/cross_view_transformers/eval{i:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1

    print(metric.compute())



if __name__ == '__main__':
    main()
