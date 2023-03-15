# DISeR: Designing Imaging Systems with Reinforcement Learning

Codebase for submission #11360. Code will be cleaned up and made publicly available upon acceptance.

## Stereo Depth

All code for stereo depth estimation can be found in the stereo_depth directory. PyRedner is a depedency of this code.

To run training:

```
python env.py
```

## Autonomous Vehicle Rig Design

All code for AV rig design can be found in the av_rig_design directory. CARLA is a dependency of this code. It must be installed and the server must be running per the CARLA documentation. The CARLA port and traffic manager port can be specified in the config.

To run training:

```
python env.py configs/expt.yaml
```
