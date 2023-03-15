import os
from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

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


log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'

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
    with torch.no_grad():
        for i, batch in enumerate(loader):
            #print("Processing batch {}!".format(i))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch)

            metric.update(pred,batch)

    print(metric.compute())



if __name__ == '__main__':
    main()
