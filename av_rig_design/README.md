# Automating AV Camera Rig Design

The code in this repo is used for training an RL agent to design an autonomous vehicle (AV) camera rig. Data from each candidate camera rig is simulated in [CARLA](https://carla.org/). The perception model, implemented as [Cross View Transformers](https://github.com/bradyz/cross_view_transformers), is jointly trained on the task of bird's eye view segmentation.

## Autonomous Vehicle Rig Design

All code for AV rig design can be found in this directory. CARLA is a dependency of this code. It must be installed and the server must be running per the CARLA documentation. The CARLA port and traffic manager port can be specified in the config.

To run training:

```
python env.py configs/expt.yaml
```
## Acknowledgements

We'd like to thank [Jonah Philion](https://www.cs.toronto.edu/~jphilion/) for the initial version of the CARLA rendering code. We'd also like to thank the authors of [Cross View Transformers](https://github.com/bradyz/cross_view_transformers) for their great work!
