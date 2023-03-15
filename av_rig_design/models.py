import os
import matplotlib.pyplot as plt
from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn
import torchvision

from cross_view_transformer.common import load_backbone
from cross_view_transformers.cross_view_transformer.model.cvt import CrossViewTransformer
from cross_view_transformers.cross_view_transformer.model.decoder import Decoder
from cross_view_transformers.cross_view_transformer.model.encoder import Encoder
from cross_view_transformer.model.backbones.efficientnet import EfficientNetExtractor
from cross_view_transformers.cross_view_transformer.data.tools import gen_dx_bx 

from stablebaselines3.common.torch_layers import BaseFeaturesExtractor
from stablebaselines3 import PPO

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

backbone = EfficientNetExtractor(
        ['reduction_2', 'reduction_4'],
        224, 400,
        'efficientnet-b4'
)

# Encoder
cross_view = {
    'heads': 4,
    'dim_head': 32,
    'qkv_bias': True,
    'skip': True,
    'no_image_features': False,
    'image_height': 224,
    'image_width': 400,
}

bev_embedding = {
    'sigma': 1.0,
    'bev_height': 200,
    'bev_width': 200,
    'h_meters': 100.0,
    'w_meters': 100.0,
    'offset': 0.0,
    'decoder_blocks': [128, 128, 64],
}

# Decoder
dim = 128
blocks = [128, 128, 64]
residual = True
factor = 2

# CVT
outputs = {
    "bev": [0, 1],
    "center": [1, 2],
}
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}
dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()

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

class BEVModel(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        self.device = device
        self.network = load_backbone(ckpt_path)
        self.network = self.network.to(device)

    def forward(self, batch):
        for name, item in batch.items():
            if name == "cameras": continue
            batch[name] = th.tensor(item).to(self.device).float()#.unsqueeze(0)
            if name in ["intrinsics", "extrinsics"]:
                batch[name] = batch[name].unsqueeze(0)
            elif name == "image" and len(batch[name].shape) == 4:
                batch[name] = batch[name].unsqueeze(0)

        """
        print("BEV Forward")
        print(batch["image"].shape)
        print(batch["extrinsics"].shape)
        print(batch["intrinsics"].shape)
        """

        #batch["image"] = batch["image"][0]
        batch["intrinsics"] = batch["intrinsics"][0]
        batch["extrinsics"] = batch["extrinsics"][0]

        """
        print(batch["image"].shape)
        print(batch["extrinsics"].shape)
        print(batch["intrinsics"].shape)
        """

        cams = batch["cameras"][0]
        batch["image"] = batch["image"][:,:cams,:,:,:]
        batch["intrinsics"] = batch["intrinsics"][:,:cams,:,:]
        batch["extrinsics"] = batch["extrinsics"][:,:cams,:,:]

        """
        print(batch["image"].shape)
        irint(fov)
        print(batch["extrinsics"].shape)
        print(batch["intrinsics"].shape)
        """


        out = self.network(batch)
        return out

    def viz(self, batch, binimgs, pred, i):
        imgs = batch['image'].detach().cpu().numpy()

        out = pred.sigmoid().cpu()

        val = 0.01
        final_dim=(128, 352)
        fH, fW = final_dim
        fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
        gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        bx_1 = bx[:2]
        dx_1 = dx[:2]

        for si in range(imgs.shape[0]):
            plt.clf()
            for imgi, img in enumerate(imgs[si]):
                ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                showimg = np.moveaxis(img,0,2) #* 255.0 
                # flip the bottom images
                #if imgi > 2:
                #    showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
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
            if th.is_tensor(out):
                out = out.detach().cpu().numpy()
            plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
            plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Oranges', alpha=0.3)

            plt.xlim((out.shape[3], 0))
            plt.ylim((0, out.shape[3]))
            add_ego(bx_1, dx_1)

            path = os.getcwd()
            imname = f'{path}/eval{i:06}_{si:03}.jpg'
            print('saving', imname)
            plt.savefig(imname)

class BEVFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, ckpt_path = None):
        super().__init__(observation_space, features_dim)
        self.dim = features_dim
        self.extractor = Encoder(
                backbone,
                cross_view,
                bev_embedding
        )
        if ckpt_path:
            """ load encoder weights only """
            ckpt = th.load(ckpt_path)["state_dict"]
            state = dict()
            for name, param in ckpt.items():
                if "backbone.encoder." in name:
                    state[name.replace("backbone.encoder.","")] = param
                elif "encoder." in name:
                    state[name.replace("encoder.","")] = param
            self.extractor.load_state_dict(state, strict=True)

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(128, 128, 1, stride=2)
        self.conv2 = nn.Conv2d(128, 128, 1, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 1, stride=2)

    def forward(self, observations):
        """
        returns:
        - z_ac: latent dim for actor critic networks (128)
        - z_bev: latent dim for perception network (128 x 25 x 25)
        """

        #for name, param in self.extractor.named_parameters():
        #    if name == "layers.0.0.conv1.weight":
        #        print("Norm: {}".format(th.norm(param)))

        #print("Feature Extractor Forward")
        """
        print(observations["image"].shape)
        print(observations["intrinsics"].shape)
        print(observations["extrinsics"].shape)
        print(observations["cameras"].shape)
        """
        observations["image"] = observations["image"].squeeze(1) #[0]
        observations["intrinsics"] = observations["intrinsics"].squeeze(1) #[0]
        observations["extrinsics"] = observations["extrinsics"].squeeze(1) #[0]

        # padding trick
        cams = int(observations["cameras"][0])
        observations["image"] = observations["image"][:,:cams,:,:,:]
        observations["intrinsics"] = observations["intrinsics"][:,:cams,:,:]
        observations["extrinsics"] = observations["extrinsics"][:,:cams,:,:]

        #"""
        #print(observations["image"].shape)
        #print(observations["intrinsics"].shape)
        #print(observations["extrinsics"].shape)
        #print(observations["cameras"].shape)
        #"""

        """
        images = []
        intrins = []
        extrins = []
        for i in range(observations["image"].shape[0]):
            images.append(observations["image"][i])
            intrins.append(observations["intrinsics"][i])
            extrins.append(observations["extrinsics"][i])
            #print(observations["image"][i].shape)
        observations["image"] = th.cat(images,0).unsqueeze(0)
        observations["intrinsics"] = th.cat(intrins,0).unsqueeze(0)
        observations["extrinsics"] = th.cat(extrins,0).unsqueeze(0)
        """
        """
        print(observations["image"].shape)
        print(observations["intrinsics"].shape)
        print(observations["extrinsics"].shape)
        """

        # forward pass
        batch = observations["image"].shape[0]
        z_bev = self.extractor(observations)
        z_ac = self.conv1(z_bev)
        z_ac = self.relu(z_ac)
        z_ac = self.conv2(z_ac)
        z_ac = self.relu(z_ac)
        z_ac = self.conv3(z_ac)
        z_ac = z_ac.reshape(batch, self.dim)

        #z_ac = th.sum(z_ac, 0, keepdim=True)
        #print("LATENT CODE SHAPE")
        #print(z_ac.shape)

        return z_ac

if __name__ == "__main__":
    fx = BEVFeatureExtractor(None, 128)
    image = th.ones((16, 1, 3, 224, 480)).float()
    I_inv = th.randn((16,1,3,3))
    E_inv = th.randn((16,1,4,4))
    batch = {'image': image,
             'intrinsics': I_inv,
             'extrinsics': E_inv,
                }
    z_ac, z_bev = fx(batch)
    print(z_ac.shape)
    print(z_bev.shape)
    exit()
