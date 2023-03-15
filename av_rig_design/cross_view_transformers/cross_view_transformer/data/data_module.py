import torch
#import pytorch_lightning as pl

from . import get_dataset_module_by_name
from .carla_dataset import compile_data


#class DataModule(pl.LightningDataModule):
class DataModule():
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.data_config = dict(data_config)
        self.loader_config = loader_config
        self.dataroot = data_config['dataset_dir']


    def get_split(self, split, loader=True, shuffle=False):
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
        data_aug_conf = {
                        'H': 224, 'W': 400,
                        #'H': 256, 'W': 256,
                        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                        'Ncams': 6,
                    }
        dataroot = self.dataroot #+ "/town3/"
        limit = None
        multi = False
        random_angle = self.data_config["random_angle"]
        if split == 'val':
            #dataroot = "/".join(dataroot.split("/")[:-2]) + "/town05/"
            #dataroot = '/data/town05/'  #'/val/'
            limit = 5000
        elif self.data_config["multi"] != "none":
            ds = self.data_config["multi"].split("_")
            multi = []
            for d in ds:
                d = d.split("-")
                path = d[0]
                p = d[1]
                if p == "":
                    p = -1 * int(d[2])
                else:
                    p = int(p)
                d = [path, p, int(d[-1])]
                multi.append(d)

        pitch_adjust = int(self.data_config["pitch_adjust"])
        noadjust = self.data_config["noadjust"]
        dataset = compile_data(dataroot, data_aug_conf=data_aug_conf,
                               grid_conf=grid_conf, parser_name='segmentationdata',pitch_adjust=pitch_adjust,limit=limit,
                               noadjust=noadjust,multi=multi,random_angle=random_angle)
        print("Final dataset length: {}".format(len(dataset)))

        if not loader:
            return dataset

        loader_config = dict(self.loader_config)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', loader=True, shuffle=False)
