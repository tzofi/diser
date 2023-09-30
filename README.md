# DISeR: Designing Imaging Systems with Reinforcement Learning

Our initial code is public! A cleaned version with improved documentation will be released by the end of __October 2023__. Stay tuned!

> __DISeR: Designing Imaging Systems with Reinforcement Learning__  
> [Tzofi Klinghoffer*](https://tzofi.github.io/), [Kushagra Tiwary*](https://www.media.mit.edu/people/ktiwary/overview/), [Nikhil Behari](https://www.media.mit.edu/people/nbehari/overview/), [Bhavya Agrawalla](https://scholar.google.com/citations?user=TdJ4Rk4AAAAJ&hl), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)  
> _International Conference on Computer Vision (_ICCV_), 2023_  
> __[Project page](https://tzofi.github.io/diser)&nbsp;/ [Paper](https://tzofi.github.io/diser/assets/tzofi2023diser.pdf)&nbsp;/ [BibTeX](https://tzofi.github.io/diser/assets/tzofi2023diser.bib)__

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
