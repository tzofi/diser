import torch
import os

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import fsspec

from stablebaselines3 import TD3
from stablebaselines3.common.monitor import Monitor
from stablebaselines3.common.results_plotter import load_results, ts2xy
from stablebaselines3.common.noise import NormalActionNoise
from stablebaselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, update: bool, hparams=None, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.perc_path = os.path.join(log_dir, 'best_perception_model')
        self.best_mean_reward = -np.inf
        self.update = update
        self.hparams = hparams

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.perc_path is not None:
            os.makedirs(self.perc_path, exist_ok=True)

    def _on_training_start(self) -> None:
        self.envs = self.training_env

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          print("------------- Callback for saving a model at {} -------------".format(self.n_calls))
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              torch.cuda.empty_cache()

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)
                  # self.perception_net.save()


                  if self.update:
                      try:

                          nets = self.envs.get_attr('network_params')
                          opt = self.envs.get_attr("optimizer")
                          sched = self.envs.get_attr("scheduler")
                          if not isinstance(nets, list):
                              nets = [nets]
                              opt = [opt]
                              sched = [sched]
                          for i in range(len(nets)):
                              save_dict = {'state_dict': nets[i],
                                           'optimizer_states': [opt[i].state_dict()],
                                           'lr_schedulers': [sched[i].state_dict()],
                                           'hyper_parameters': self.hparams,
                              }

                              bytesbuffer = io.BytesIO()
                              torch.save(save_dict, bytesbuffer)
                              with fsspec.open(os.path.join(self.perc_path, "perception_{}.ckpt".format(i)), "wb") as f:
                                f.write(bytesbuffer.getvalue())

                              #torch.save(save_dict, "{}/env_{}_perception.ckpt".format(self.perc_path, i, pickle_module=pickle))
                              print("Saving Perception Model at: {}/perception_{}.ckpt".format(self.perc_path, i))

                      except Exception as e:
                          print(e)

        return True
