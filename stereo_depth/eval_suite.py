import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import json
import warnings
import triangulation_chip_placement_extrinsics_rewardBtrain
import triangulation_chip_placement_extrinsics_train_on_reset
import triangulation_chip_placement_extrinsics_fov
import tqdm

from time import time
from PIL import Image 
from pathlib import Path
from dotmap import DotMap
from datetime import datetime

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from models import NatureCNN, SimpleViTFlexInput, PerceptionNetExtrinsics, get_ViT, PolicyFeatureExtractorExtrinsics
from plot import plot_results

def load_env_config_dict(logdir):
    file = "{}/exp_info.json".format(logdir)
    with open(file) as f: 
        d = json.load(f) 
        keys = list(d.keys())
        keys.sort()
        env = d[keys[-1]]['env']
        config = DotMap(env)

    return config

def get_camera_params():
    params = {}
    params['count'] = 0
    params['l1_loss'] = []
    params['std_l1_loss'] = []
    params['rewards'] = []
    params['std_rewards'] = []
    
    params['x'] = []
    params['mean_x'] = 0.
    params['std_x'] =  0.

    params['z'] = []
    params['mean_z'] = 0.
    params['std_z'] = 0.

    params['theta'] = []
    params['mean_theta'] = 0.
    params['std_theta'] = 0.

    params['sum_of_rewards'] = []
    params['mean_rewards'] = []
    params['std_rewards'] = []
    params['std_l1'] = []
    params['coverage'] = []
    
    # add all info
    params['camera_array'] = [] # will be a list of lists
    return params

def parse_observation(obs):
    images = torch.tensor(obs['images'])
    nc = obs['cameras']
    images = images[:,0:nc,:,:]
    count = 0 
    for n in range(nc): 
        img = images[:,n,:,:]
        if (img.numpy() > 0).any():
            count += 1

    return nc, count 

def add_to_processed_rollouts(count, info, rewards, cameras, l1_loss):

    if len(cameras) not in info.keys():
        info[len(cameras)] = get_camera_params()

    info[len(cameras)]['count'] += 1
    info[len(cameras)]['coverage'].append(count)
    sum_of_reward = np.sum(np.array(rewards))

    info[len(cameras)]['sum_of_rewards'].append(sum_of_reward)
    info[len(cameras)]['mean_rewards'].append(np.array(info[len(cameras)]['sum_of_rewards']).mean())
    info[len(cameras)]['std_rewards'].append(np.array(info[len(cameras)]['sum_of_rewards']).std())
    
    info[len(cameras)]['l1_loss'].append(l1_loss[-1])
    info[len(cameras)]['std_l1'].append(np.array(info[len(cameras)]['l1_loss']).std())

    info[len(cameras)]['l1_loss'].append(l1_loss[-1])
    info[len(cameras)]['rewards'].append(rewards)

    info[len(cameras)]['camera_array'].append(cameras)

    xs = []
    zs = []
    thetas = []
    for i in range (len(cameras)):
        xs.append(cameras[i][0])
        zs.append(cameras[i][1])
        thetas.append(cameras[i][2])

    info[len(cameras)]['x'].append(xs)
    info[len(cameras)]['mean_x'] = np.array(info[len(cameras)]['x']).mean(0)
    info[len(cameras)]['std_x'] = np.array(info[len(cameras)]['x']).std(0)

    
    info[len(cameras)]['z'].append(zs)
    info[len(cameras)]['mean_z'] = np.array(info[len(cameras)]['z']).mean(0)
    info[len(cameras)]['std_z'] = np.array(info[len(cameras)]['z']).std(0)

    info[len(cameras)]['theta'].append(thetas)
    info[len(cameras)]['mean_theta'] = np.array(info[len(cameras)]['theta']).mean(0)
    info[len(cameras)]['std_theta'] = np.array(info[len(cameras)]['theta']).std(0)

    return info

def add_to_raw_rollouts(count, rewards, l1_loss, cameras):
    raw_rollout_info = {}
    raw_rollout_info['rewards'] = np.array(rewards)
    raw_rollout_info['sum_reward'] = np.sum(np.array(rewards))
    raw_rollout_info['l1_loss'] = l1_loss[-1]
    raw_rollout_info['l1_loss_array'] = np.array(l1_loss)
    raw_rollout_info['num_cameras'] = len(cameras)
    raw_rollout_info['coverage']  = count
    raw_rollout_info['cameras'] = cameras
    return raw_rollout_info

def load_policy_test_suite(logdir, perception_ckpt, set_sphere_pos=None, 
                            set_radius=None, set_sphere_range=None, 
                            new_scene_every_episode=True, env_str='rewardBtrain', 
                            return_config=False):
    file = "{}/exp_info.json".format(logdir)
    with open(file) as f: 
        d = json.load(f) 
        keys = list(d.keys())
        keys.sort()
        env = d[keys[-1]]['env']
        config = DotMap(env)

    if set_sphere_pos is not None: 
        config.set_gt_sphere = set_sphere_pos
    
    if set_radius is not None: 
        config.set_radius = set_radius

    if set_sphere_range is not None: 
        print("Setting Sphere Range to", set_sphere_range)
        config.sphere_z = set_sphere_range

    # Load Perception Net 
    perception_ckpt = "{}/best_model/{}".format(logdir, perception_ckpt)
    pnet = PerceptionNetExtrinsics(image_size=128)
    pnet.load_state_dict(torch.load(perception_ckpt))
    pnet.eval()
    
    # load environmnet 
    if env_str == 'rewardBtrain':
        env = triangulation_chip_placement_extrinsics_rewardBtrain.RednerEnv("0", env_str="mono", config=config, 
                                                                     perception_net=pnet, 
                                                                     new_scene_every_episode=new_scene_every_episode, 
                                                                     train_network=False)
    elif env_str == 'train_on_reset':
        env = triangulation_chip_placement_extrinsics_train_on_reset.RednerEnv("0", env_str="mono", config=config, 
                                                                         perception_net=pnet, 
                                                                         new_scene_every_episode=new_scene_every_episode, 
                                                                         train_network=False)
    elif env_str == 'fov':
        env = triangulation_chip_placement_extrinsics_fov.RednerEnv("0", env_str="mono", config=config, 
                                                                         perception_net=pnet, 
                                                                         new_scene_every_episode=new_scene_every_episode, 
                                                                         train_network=False)
    else:
        raise ValueError("{} Not found".format(env_str))

    # load PPO Model 
    ppo_path = "{}/best_model.zip".format(logdir)
    model = PPO.load(ppo_path, env)
    if return_config: 
        return model, env, config
    return model, env 

def generate_n_cam_heatmap(processed_info, action_space, key=2):

    x_space = action_space[1][1]*2
    z_space = action_space[2][1] - action_space[2][0]
    def clip(x, _shift, _min, _max):
        x =  x + _shift
        x = np.maximum(_min, x)
        x = np.minimum(_max, x)
        return x
    images = {}
    images['max_baselines'] = []
    images['avg_baselines'] = []
    for i in range (key):
        # for each camera create a heatmap
        images[i+1] = np.zeros((int(z_space)+1, int(x_space)+1)) 
    try: 
        cameras = processed_info[key]['camera_array']
    except KeyError as e: 
        print("No info found with {} num cams".format(key))
        return images

    # print(cameras)
    images['baselines'] = []
    for list_of_cam in cameras:
        if len(list_of_cam) != key : 
            raise ValueError(key, list_of_cam)
        for i in range(len(list_of_cam)):
            cam = list_of_cam[i]
            x = cam[0]
            z = cam[1]
            thetas = cam[2]
            x = clip(x, action_space[1][1], 0, x_space)
            z = clip(z, -1.0 * action_space[2][0], 0, z_space)
            images[i+1][int(z), int(x)] +=1 
        if key == 2: 
            # get baseline for two cams 
            baseline =  np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[1][:2])**2))
            images['max_baselines'].append(baseline)
        if key == 3: 
            b1 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[1][:2])**2))
            b2 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[2][:2])**2))
            b3 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[2][:2])**2))
            images['max_baselines'].append(np.max([b1,b2,b3]))
            images['avg_baselines'].append(np.mean([b1,b2,b3]))
        if key == 4: 
            b1 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[1][:2])**2))
            b2 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[2][:2])**2))
            b3 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[3][:2])**2))
            b4 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[2][:2])**2))
            b5 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[3][:2])**2))
            b6 = np.sqrt(np.abs(np.array(list_of_cam[2][:2])**2 - np.array(list_of_cam[3][:2])**2))
            images['max_baselines'].append(np.max([b1,b2,b3,b4, b5,b6]))
            images['avg_baselines'].append(np.mean([b1,b2,b3,b4, b5,b6]))
        if key == 5: 
            b1 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[1][:2])**2))
            b2 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[2][:2])**2))
            b3 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[3][:2])**2))
            b4 = np.sqrt(np.abs(np.array(list_of_cam[0][:2])**2 - np.array(list_of_cam[4][:2])**2))
            b5 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[2][:2])**2))
            b6 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[3][:2])**2))
            b7 = np.sqrt(np.abs(np.array(list_of_cam[1][:2])**2 - np.array(list_of_cam[4][:2])**2))
            b8 = np.sqrt(np.abs(np.array(list_of_cam[2][:2])**2 - np.array(list_of_cam[3][:2])**2))
            b9 = np.sqrt(np.abs(np.array(list_of_cam[2][:2])**2 - np.array(list_of_cam[4][:2])**2))
            b11= np.sqrt(np.abs(np.array(list_of_cam[3][:2])**2 - np.array(list_of_cam[4][:2])**2))
            images['max_baselines'].append(np.max([b1,b2,b3,b4,b5,b6,b7,b8,b9,b11]))
            images['avg_baselines'].append(np.mean([b1,b2,b3,b4,b5,b6,b7,b8,b9,b11]))
    
    # f, axarr = plt.subplots(1,key)
    # for i in range(key):
    #     axarr[0,key].imshow(images[key])
    #     axarr[0,key].set_title("Distribution for Camera {}/{}".format(i, key))

    return images

def generate_camera_placement_heatmap(processed_info, action_space):

    x_space = action_space[1][1]*2
    z_space = action_space[2][1] - action_space[2][0]
    def clip(x, _shift, _min, _max):
        x =  x + _shift
        x = np.maximum(_min, x)
        x = np.minimum(_max, x)
        return x
    images = {}
    for key in processed_info.keys(): 
        cameras = processed_info[key]['camera_array']
        images[key] = np.zeros((int(z_space)+1, int(x_space)+1))
        # print(cameras)
        for list_of_cam in cameras:
            if len(list_of_cam) == 0 : 
                raise ValueError(cameras, list_of_cam)
            for cam in list_of_cam:
                x = cam[0]
                z = cam[1]
                thetas = cam[2]
                x = clip(x, action_space[1][1], 0, x_space)
                z = clip(z, -1.0 * action_space[2][0], 0, z_space)
                images[key][int(z), int(x)] +=1 
    return images

def generate_all_camera_placement_heatmap(raw_rollouts, action_space):

    x_space = action_space[1][1]*2
    z_space = action_space[2][1] - action_space[2][0]
    def clip(x, _shift, _min, _max):
        x =  x + _shift
        x = np.maximum(_min, x)
        x = np.minimum(_max, x)
        return x

    image = np.zeros((int(z_space)+1, int(x_space)+1))
    for rollout in raw_rollouts: 
        cameras = rollout['cameras']
        for cam in cameras:
            x = cam[0]
            z = cam[1]
            thetas = cam[2]
            x = clip(x, action_space[1][1], 0, x_space)
            z = clip(z, -1.0 * action_space[2][0], 0, z_space)
            image[int(z), int(x)] +=1 
    
    return {"all_cameras": image} 

def get_coverage_stats(raw_rollouts, print_stats=True):
    
    coverage = {}
    for rollout in raw_rollouts:
        count = rollout['coverage']
        if count not in coverage.keys(): 
            coverage[count] = {}
            coverage[count]['num_cameras'] = []
            coverage[count]['l1'] = []

        coverage[count]['num_cameras'].append(rollout['num_cameras'])
        coverage[count]['l1'].append(rollout['l1_loss'])
    
    for key in coverage.keys(): 
        coverage[key]['mean_loss'] = np.array(coverage[key]['l1']).mean()
        coverage[key]['std_loss'] = np.array(coverage[key]['l1']).std()
        coverage[key]['mean_num_cameras'] = np.array(coverage[key]['num_cameras']).mean()
        coverage[key]['std_num_cameras'] = np.array(coverage[key]['num_cameras']).std()

        if print_stats: 
            print('-------- When Coverage is {} --------'.format(key))
            print('mean_loss -> {}'.format(coverage[key]['mean_loss']))
            print('std_loss -> {}'.format(coverage[key]['std_loss']))
            print('mean_num_cameras -> {}'.format(coverage[key]['mean_num_cameras']))
            print('std_num_cameras -> {}'.format(coverage[key]['std_num_cameras']))

    return coverage

def print_stats(info, num_trials):
    
    # print("------------ PRINTING STATS ----------------")
    # sort it by keys 
    info = {k: info[k] for k in sorted(info.keys())}
    print("Total Number of Trials: ", num_trials)
    for key in info.keys():
        count = info[key]['count']
        print("----------- Number of Cameras: {} -----------".format(key))
        l1_loss = np.mean(info[key]['l1_loss'])
        std_l1 = np.mean(info[key]['std_l1'])
        reward =  np.mean(info[key]['sum_of_rewards'])
        std_reward =  np.mean(info[key]['std_rewards'])
        mean_x = info[key]['mean_x']
        std_x = info[key]['std_x']
        
        mean_z = info[key]['mean_z']
        std_z = info[key]['std_z']

        mean_theta = info[key]['mean_theta']
        std_theta = info[key]['std_theta']

        coverage = np.array(info[key]['coverage'])
        mean_coverage = coverage.mean()
        std_coverage = coverage.std()
        
        percent = 100. * count/num_trials
        print("Percentage {} %% ({}/{}".format(percent, count, num_trials))
        print("Average reward of {} with std {}.".format(reward, std_reward))
        print("L1 Loss of {} with std {}.".format(l1_loss, std_l1))
        print("Mean location of x: {} with std {}.".format(mean_x, std_x))
        print("Mean location of z: {} with std {}.".format(mean_z, std_z))
        print("Mean location of theta: {} with std {}.".format(mean_theta, std_theta))
        print("Mean Coverage: {} with std {}.".format(mean_coverage, std_coverage))
    # print("------------ PRINTING STATS ----------------")
    return info 


def collect_rollouts(model, env, print_info=True, num_trials=10):
    
    print("Starting Rollouts for {} Trials...".format(num_trials))
    processed_info = {}

    raw_rollouts = []
    for i in tqdm.tqdm(range(num_trials)):
        obs = env.reset()
        rewards = []
        l1_losses = []
        for i in range (env.rollout_length):
            action = model.predict(obs)
            obs, reward, done, info = env.step(action[0])
            rewards.append(reward)
            if 'episode_loss' in info:
                l1_losses.append(info['episode_loss'])
            if done: 
                nc, count = parse_observation(obs)
                raw_rollout_info = add_to_raw_rollouts(count, rewards, l1_losses, env.cameras)
                raw_rollouts.append(raw_rollout_info)
                processed_info = add_to_processed_rollouts(count, processed_info, rewards, env.cameras, l1_losses)
                break
    
    if print_info:
        print_stats(processed_info, num_trials)

    return processed_info, raw_rollouts


if __name__ ==  "__main__":
    print("_____________________Running Eval Suite_____________________")
    perception_ckpt = "env_0_best_perception.ckpt"
    # logdir = "logs-extrinsics-feb24/large-state-space/exp1_scratch_large_z/"
    logdirs = ["logs-extrinsics-feb24/large-state-space/exp1_scratch/", 
               "logs-extrinsics-feb24/large-state-space/exp1_scratch_large_z"]
    logdirs = ["logs-extrinsics-mar2/optimizer-fix/exp1_v_large_z_30fov", "logs-extrinsics-mar2/optimizer-fix/exp1_v_large_z_45fov"]

    NUM_TRIALS= 7000
    for logdir in logdirs: 
        model, env = load_policy_test_suite(logdir, perception_ckpt, 
                                                    new_scene_every_episode=True, env_str='rewardBtrain')
        processed_info, raw_rollouts = collect_rollouts(model, env, print_info=True, num_trials=NUM_TRIALS)
        results = {}
        results['NUM_TRIALS'] = NUM_TRIALS
        results['processed_info'] = processed_info
        results['raw_rollouts'] = raw_rollouts

        np.save('{}/results_random_sphere_5ktrials_with_coveage.npy'.format(logdir), results)
