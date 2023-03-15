import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os 

from datetime import datetime
import torch
import torch.nn as nn
# from stable_baselines.common import set_global_seeds
from PIL import Image
import triangulation_scene
from redner_scene import materials
from time import time
from torch.optim.lr_scheduler import StepLR
from rl_utils import SaveOnBestTrainingRewardCallback
from copy import deepcopy
from PIL import Image 
from pathlib import Path
from dotmap import DotMap
import json

from datetime import datetime
import gym
from gym import spaces
import warnings
import stereo_triangulation

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

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_env(rank, run_id, res, config, env_str="complex", perception_net=None, optimizer=None, scheduler=None, new_scene_every_episode=False, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    logdir = "./tmp/"
    def _init():
        env = RednerEnv(run_id=run_id, res=res, env_str=env_str, config=config,
                        perception_net=perception_net, optimizer=optimizer, scheduler=scheduler, 
                        new_scene_every_episode=new_scene_every_episode)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

class RednerEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, run_id, config, perception_net=None, optimizer=None, scheduler=None, 
                 env_str= "complex", res=(128,128), loss_type="depth", 
                 new_scene_every_episode=False, train_network=True):
        """
        obs spae: (image, intrinsics, extrinsics, cameras) 
        Action Space: (c, x \in 1D ,z \in 1D ,\theta) 
        """
        super(RednerEnv, self).__init__()

        self.config = config
        self.env_str = env_str
        self.new_scene_every_episode = new_scene_every_episode
        self.res = res
        self.tri = None
        self.sphere_x = None
        self.sphere_z = None
        self.light_pos = None

        self.train_network = train_network
        print("------------------------------------------")
        print("Training Perception Network set to {}".format(self.train_network))
        print("------------------------------------------")
        
        self.cameras = []
        self.max_images = 6 
        self.steps = 0
        self.extrinsics_shape = (self.max_images, 9)
        self.image_shape = (3, self.max_images, self.res[0], self.res[1])
        self.image_shape_2 = (1, 3, self.max_images, self.res[0], self.res[1])
                                            
        self.observation_space = spaces.Dict(
            spaces={
                "images": spaces.Box(0, 255, self.image_shape, dtype=np.uint8),
                "cameras": spaces.Discrete(self.max_images  + 1),
                "extrinsics": spaces.Box(-np.inf, np.inf, self.extrinsics_shape, dtype=np.float32)
                # "cameras": spaces.Box(0,self.max_images, (1,1), dtype=np.uint8),
            }
        )

        self.action_space = spaces.Box(low=np.array([-1., -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32) 

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        if 'hfov' in self.config:
            self.hfov = self.config.hfov
        else:
            HFOV = 50.
            print("NOTE: SETTING THE HFOV TO {}".format(HFOV))
            self.hfov = HFOV

        print("hfov is set to", self.hfov)

        self.setup_scene()
        self.compute_gt_depth = lambda cam_z: np.abs(cam_z - self.gt_sphere_pos[1] - self.radius)

        self.rollout_length = 6

        self.cam_thresh = 0.5
        self._LOOK_AT_CONSTANT = 10.

        self.loss_type = loss_type

        # Depth estimation MLP (takes image as input, predicts depth of object)
        # self.model = SimpleViTFlexInput(image_size=self.res).to(self.device)
        self.model = perception_net
        self.model.to(self.device)

        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        if optimizer is None: 
            print("Instantiating a new optimizer...")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-4)
        else:
            print("Using provided instantiator...")
            self.optimizer = optimizer 

        if scheduler is None: 
            print("Instantiating a new  scheduler...")
            self.scheduler = StepLR(self.optimizer, step_size=100)
        else:
            print("Using provided scheduler...")
            self.scheduler = scheduler 

        self.perception_dataset = {
            'obs': [],
            'depth': [],
            'extrinsics': []
        }

        self.rewards = []

    def setup_scene(self):

        if self.env_str == 'complex':
            print("Using Complex Env. ")
            self.tri = triangulation_scene.TriangulationSceneBounded()
            self.tri.prepare_base_scene()
            self.action_ranges = self.config.action_ranges
            self.radius = np.random.randint(3,9)

            self.gt_pos= 0.0

            if self.sphere_x is None or self.new_scene_every_episode:
                self.sphere_x = np.random.uniform(2., 4.) #-30,31)
                self.sphere_z = np.random.uniform(0., 31.)

            self.shapes = self.tri.render_sphere(position=[self.gt_pos, 5.,15.], radius=self.radius, material_id="m_plastic")
            self.area_light = self.tri.illuminate_scene(torch.tensor([-20., 5., 50.]).cuda(), torch.tensor(-180.))
            self.gt_sphere_pos = (self.gt_pos, 15)

        elif self.env_str == 'mono':
            print("Using enviornment with no monocular cues. ")
            self.tri = stereo_triangulation.TriangulationSceneBounded()
            self.tri.prepare_base_scene(bound_scene=False)
            self.light_pos = torch.tensor([0,100,3])
            self.action_ranges = self.config.action_range
            self.radius = np.random.randint(3,9)

            if self.sphere_x is None or self.new_scene_every_episode: 
                self.sphere_x = np.random.uniform(self.config.sphere_x[0], self.config.sphere_x[1]) #-30,31)
                self.sphere_z = np.random.uniform(self.config.sphere_z[0], self.config.sphere_z[1])

            # print("Resetting Sphere to: ({},{})".format(self.sphere_x, self.sphere_z))
            
            self.gt_sphere_pos = (self.sphere_x, self.sphere_z)
            
            if 'set_gt_sphere' in self.config:
                print("Hard Coding Sphere Position to {}".format(self.config.set_gt_sphere))
                self.gt_sphere_pos = self.config.set_gt_sphere

            if 'set_radius' in self.config:
                print("Hard Coding Sphere Radius to {}".format(self.config.set_radius))
                self.radius = self.config.set_radius

            self.shapes = self.tri.render_sphere(position=[self.sphere_x, 5., self.sphere_z], radius=self.radius, material_id="m_plastic")
            self.area_light = self.tri.illuminate_scene(torch.tensor([0., 100., 3.]).cuda(), torch.tensor(-180.))

    def predict_depth(self, obs, gt_depth, extrinsics):
        with torch.no_grad():
            self.model.eval()
            pred_depth = self.model(obs.to(self.device).float(), extrinsics.to(self.device).float()).detach()
            # pred_depth = self.model(deepcopy(obs).to(self.device).float().detach(), deepcopy(extrinsics).to(self.device).float().detach())
            episode_loss = self.criterion(pred_depth.detach(), gt_depth.to(self.device).float())
            episode_loss = torch.abs(episode_loss).detach().item()

            # some good old reward hacking 
            e_min = 0
            e_max = gt_depth.item()
            reward = -1 * (2*(np.minimum(episode_loss, e_max) - e_min) / (e_max - e_min) - 1) # scale it to [-1,1]
            if np.random.uniform() > 0.9: 
                print("Randomly printing Model Eval: predicted depth {} , gt depth {} with episode loss {}".format(pred_depth, gt_depth, episode_loss))

        return reward, episode_loss, pred_depth

    def train_perception_network(self, obs, gt_depth, extrinsics):
        # train after calculating the reward... 
        total_loss = 0.
        self.model.train()
        # add obs to dataset
        self.perception_dataset['obs'].append(obs) # should be 
        self.perception_dataset['extrinsics'].append(extrinsics)
        self.perception_dataset['depth'].append(gt_depth)

        if len(self.perception_dataset['obs']) > self.config.max_prev_datasets: 
            self.perception_dataset['obs'].pop(0)
            self.perception_dataset['extrinsics'].pop(0)
            self.perception_dataset['depth'].pop(0)
        
        # train network 
        gt_depth_batch = torch.stack(self.perception_dataset['depth'], 0).to(self.device).squeeze(1).float()
        for j in range(self.config.num_epochs): 
            if True: #len(self.perception_dataset['obs']) < BATCH_SIZE: 
                pred_depth_batch = []
                self.optimizer.zero_grad()
                for i in range(len(self.perception_dataset['obs'])):
                    data = self.perception_dataset['obs'][i].to(self.device)
                    extrinsics_batch = self.perception_dataset['extrinsics'][i].to(self.device).reshape(-1, 9).float()
                    pred_depth = self.model(data, extrinsics_batch)
                    pred_depth_batch.append(pred_depth)
            
                pred_depth_batch = torch.cat(pred_depth_batch, 0)
                # print(pred_depth_batch, gt_depth_batch)
                loss = self.criterion(pred_depth_batch, gt_depth_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()


        total_loss = total_loss/self.config.num_epochs
        return total_loss


    def _rescale(self, actions):
        """
        Actions are normalized between -1 and 1, so we need to rescale
        """
        rescaled_actions = []
        min_value = -1.0
        max_value = 1.0
        # print(actions)
        for i, act in enumerate(actions):
            # print(act)
            # print(i, act[i])
            a = self.action_ranges[i][0]
            b = self.action_ranges[i][1]
            rescaled = (b-a) * (act - min_value) / (max_value - min_value) + a
            rescaled_actions.append(rescaled)
        return np.array(rescaled_actions) 

    def step(self, u):
        
        if self.config.use_random:
            # print("Randomly Sampling Actions!") 
            u = self.action_space.sample()

        if u.shape[0] == 1:
            u = u[0]

        cam_conf = u[0]
        u = self._rescale(u)
        cam_conf = u[0]

        if cam_conf > self.cam_thresh:
            self.cameras.append(tuple(u[1:]))
        
        self.cameras.sort()

        # get obs 
        obs = self._get_obs()
        losses = {"total_loss": np.inf, "episode_loss": np.inf, "pred_depth": np.inf}
        if obs is None: 
            num_cameras = 1
            obs_tensor = torch.zeros(self.image_shape) #.unsqueeze(0)
            reward = -5. # place at least 1 camera!
            self.state = {
                "images": to_uint8(obs_tensor.detach()).cpu().numpy(),
                "cameras": num_cameras,
                "extrinsics": torch.zeros((6,9)).cpu().numpy()
            }
        else:
            zs = np.array(self.cameras)[:,1]
            avg_z = np.mean(zs)
            gt_depth = float(np.abs(self.compute_gt_depth(avg_z)))

            num_cameras = obs['num_cameras']
            imgs = obs['imgs']
            obs_tensor = imgs.unsqueeze(0) # (B, 3, num_images, 128,128)
            
            # compute rewards 
            reward, episode_loss, pred_depth = self.predict_depth(obs_tensor, torch.tensor(gt_depth).reshape(1,1), obs["extrinsics"])
            if self.train_network:
                total_loss = self.train_perception_network(obs_tensor, torch.tensor(gt_depth).reshape(1,1), obs["extrinsics"])
            else:
                total_loss = np.inf 
            losses = {"total_loss": total_loss, "episode_loss": episode_loss, 'pred_depth': pred_depth}
        
            if obs_tensor.shape[2] < self.max_images: 
                add_ = self.max_images - obs_tensor.shape[2]
                num_cameras = obs_tensor.shape[2]
                add_tensor = torch.zeros((1,3, add_, obs_tensor.shape[3], obs_tensor.shape[-1]))
                obs_tensor_ = torch.concat([obs_tensor, add_tensor], dim=2).reshape(self.image_shape)

                add_tensor = torch.zeros((add_, 9))
                extrinsics = torch.concat([obs["extrinsics"], add_tensor], dim=0).reshape((-1,9))

                self.state = {
                    "images": to_uint8(obs_tensor_.detach()).cpu().numpy(),
                    "cameras": num_cameras,
                    "extrinsics": extrinsics.cpu().numpy()
                }
            else:
                num_cameras = obs_tensor.shape[2]
                self.state = {
                    "images": to_uint8(obs_tensor.reshape(self.image_shape).detach()).cpu().numpy(),
                    "cameras": num_cameras,
                    "extrinsics": obs["extrinsics"].cpu().numpy()
                }

        self.steps += 1
        done = False
        if losses['episode_loss'] < self.config.done_marker:
            print("episode loss {} < 2, we're done!".format(losses['episode_loss'])) 
            done = True

        if self.steps % (self.rollout_length-1) == 0 or (self.state["cameras"] > self.max_images):
            done = True 

        return self.state, reward, done, losses

    def reset(self):
        # print("Reseting...")
        self.steps = 0 
        self.cameras = []
        self.setup_scene()

        obs_tensor = torch.zeros(self.image_shape)
        self.state = {
                "images": to_uint8(obs_tensor.detach()).cpu().numpy(),
                "cameras": 1,
                "extrinsics": torch.zeros((self.max_images,9)).cpu().numpy()
        }
        return self.state

    def _get_obs(self, type='step'):
        """
        """
        cams = []
        cameras = []
        imgs = []
        extrinsics = []

        if type == 'reset':
            # instantiate one random camera and pass that
            x = np.random.uniform(low=self.action_ranges[1][0], high=self.action_ranges[1][1])
            z = np.random.uniform(low=self.action_ranges[2][0], high=self.action_ranges[2][1])
            theta = np.random.uniform(low=self.action_ranges[3][0], high=self.action_ranges[3][1])
            theta_rad = theta * (np.pi / 180.)

            cam, camera = self.tri.set_camera(res=self.res, 
                                            hfov=self.hfov, 
                                            look_at=[x + (self._LOOK_AT_CONSTANT * np.sin(theta_rad)), 5., z - (self._LOOK_AT_CONSTANT*np.abs(np.cos(theta_rad)))],
                                            eye_pos=[x,10.,z])
            img = self.tri.get_obs_from_agent(self.shapes, cam, camera, self.area_light, res=self.res, light_pos = self.light_pos)
            img = self.preprocess(img)
            imgs.append(img)
            cams.append(cam)
            cameras.append(camera)
            extrinsics.append(camera.camera.cpu().numpy())
            extrinsics = torch.tensor(extrinsics).reshape(-1, 9)
            num_cameras = 1
        else:
            if len(self.cameras) > 0: 
                num_cameras = len(self.cameras)
                for u in self.cameras: 
                    x = float(u[0])
                    z = float(u[1])
                    theta_rad = float(u[2]) * np.pi / 180.
                    cam, camera = self.tri.set_camera(res=self.res, 
                                                        hfov=self.hfov, 
                                                        look_at=[x + (self._LOOK_AT_CONSTANT * np.sin(theta_rad)), 5., z - (self._LOOK_AT_CONSTANT*np.abs(np.cos(theta_rad)))],
                                                        eye_pos=[x,10.,z])
                    if self.light_pos is None: 
                        img = self.tri.get_obs_from_agent(self.shapes, cam, camera, self.area_light, res=self.res)
                    else:
                        img = self.tri.get_obs_from_agent(self.shapes, cam, camera, self.area_light, res=self.res, light_pos = self.light_pos)
                    img = self.preprocess(img)
                    imgs.append(img)
                    cams.append(cam)
                    cameras.append(camera)
                    extrinsics.append(camera.camera.cpu().ravel())
                extrinsics = torch.cat(extrinsics, 0).reshape(-1, 9)
            else:
                return None

        obs = {
            "imgs": torch.tensor(np.array(imgs)).reshape(-1, 3, 128, 128).permute((1,0,2,3)), # should always be (3, num_imgs, 128, 128)
            "cams": cams,
            "cameras": cameras,
            "extrinsics": extrinsics,
            "num_cameras": num_cameras,
        }
        return obs

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            res = self.res[0]
            # print(self.state.shape, res)
            state = torch.tensor(self.state['images']).squeeze(0).permute(1,2,3, 0).numpy() # num_frames,h,w,3
            out_image = [] #np.zeros((res*2, res*3, 3)) # max of 6 images
            for i in range(state.shape[0]):
                if i == 0 : 
                    out_image = state[0]
                else:
                    out_image = np.hstack([out_image, state[i]])

            return np.array(out_image)
        else:
            raise
        

    def preprocess(self, im):
        """
        should be clipped between 0 and 1!
        """
        assert len(im.shape) == 3
        im = im.detach().cpu().numpy()
        im = np.rollaxis(im, 2, 0).astype(np.float32)#/255.0
        return im

def to_uint8(im):
    im = im * 255. 
    im = im.to(torch.uint8)
    return im

if __name__ == "__main__":
    print("_____________________Running Gym Environment_____________________")


    # ----------- ASSIGN VARIABLES -----------
    num_cpu = 1 # 6  # Number of processes to use
    num_cpu = 5 # 6  # Number of processes to usess
    # num_cpu = 15 # 6  # Number of processes to usess

    N_STEPS=256 #10 #128
    BATCH_SIZE=128 #8 # 64
    CHECK_FREQ=64 #8 # 64

    ENV_STR="mono"
    RES = (128,128)

    PPO_PATH = None
    PERCEPTION_PATH = None
    LOAD_FROM_PERVIOUS_EXP = True 
    # LOAD_FROM_PERVIOUS_EXP = False

    NUM_EPOCHS = 5 #2 #
    MAX_PREV_DATASETS = 35
    _ACTION_RANGES_ = [
                [0., 1.], # conf
                [-15., 15.], #[-30., 31.], # x SHOULD BE THE SAME FOR THE SPHERE! 
                # [-10, 10.], #[-30., 31.], # x SHOULD BE THE SAME FOR THE SPHERE! 
                # [49., 51.], # z
                [69., 80.], # z
                [-60., 60.] # theta
            ]
    # _SPHERE_X_ = (-5.,5.)
    _SPHERE_X_ = (-10.,10.)
    _SPHERE_Z_ = (1.,60.)
    _HFOV_ = 30. # small fov 
    _HFOV_ = 45. # slightly bigger fov 
    # USE_RANDOM = True
    USE_RANDOM = False

    if USE_RANDOM: 
        print("--------------- Will be RANDOMLY Sampling Actions! ---------------") 

    # ----------- ASSIGN VARIABLES -----------

    # run_id = "MOVING_sphere_exp2_donemarkernotset_cont_2"
    # run_id = "exp1_scratch"
    # run_id = "exp1_scratch_large_z_trial1"
    # run_id = "exp1_scratch_v_large_z_small_fov_2_random"
    # run_id = "exp1_v_large_z_30fov"
    run_id = "run3_exp1_v_large_z_45fov"
    # run_id = "run2_exp1_v_large_z_30fov"

    # run_id = "exp1_v_large_z_45fov_50buffer5epoch"
    log_dir = "./logs-extrinsics-mar2/optimizer-fix/{}/".format(str(run_id))
    log_dir = "./logs-extrinsics-mar2/runs/{}/".format(str(run_id))
    
    NEW_SCENE_EVERY_EPISODE = True #True

    LOAD_FROM_PPO_PATH = PPO_PATH
    LOAD_FROM_PERCEPTION_PATH = PERCEPTION_PATH #False
    vit = None
    # ------------ LOAD PERCEPTION MODEL FROM CHECKPOINT ------------
    if PERCEPTION_PATH is not None:
        print("Loading Perception Net from path: {}".format(PERCEPTION_PATH))
        perception_net = PerceptionNetExtrinsics(vit=vit, image_size=128)
        perception_net.load_state_dict(torch.load(PERCEPTION_PATH))
    else:
        perception_net = PerceptionNetExtrinsics(vit=vit, image_size=128)
    # ------------ LOAD PERCEPTION MODEL FROM CHECKPOINT ------------

    optimizer = None # torch.optim.Adam(perception_net.parameters(), lr=1.0e-5)
    scheduler = None # StepLR(optimizer, step_size=100)

    # ------------ SAVE INFO ABOUT EXPERIMENT ------------
    if not LOAD_FROM_PERVIOUS_EXP:
        print("Creating Experiment Config...")
        json_info = {}
        json_info['run_id'] = run_id
        json_info['log_dir'] = log_dir
        json_info['env'] = {}
        json_info['env']['action_range'] = _ACTION_RANGES_
        json_info['env']['new_scene_every_episode'] = NEW_SCENE_EVERY_EPISODE
        json_info['env']['sphere_x'] = _SPHERE_X_
        json_info['env']['sphere_z'] = _SPHERE_Z_
        json_info['env']['num_epochs'] = NUM_EPOCHS
        json_info['env']['max_prev_datasets'] = MAX_PREV_DATASETS
        json_info['env']['hfov'] = _HFOV_
        json_info['env']['done_marker'] = -1.0 # if -1 then done will be executed as per internal logic. 
        json_info['env']['use_random'] = USE_RANDOM # if -1 then done will be executed as per internal logic. 
        config = DotMap(json_info['env'])
        
        if PERCEPTION_PATH is None: 
            json_info['from_saved_checkpoint'] = False
        else:
            json_info['from_saved_checkpoint'] = True
            json_info['perception_path'] = PERCEPTION_PATH
            json_info['ppo_path'] = PPO_PATH

        path = Path('{}/exp_info.json'.format(log_dir))
        now = datetime.now()
        print("Starting Experiment At=", now)
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        if path.is_file():
            with open(str(path)) as json_file:
                data = json.load(json_file)
            with open(str(path), "w") as outfile:
                data[dt_string] = json_info
                json.dump(data, outfile)
        else:
            os.makedirs(log_dir, exist_ok = True)
            # path.makedirs(parents=True, exist_ok=True)
            with open(str(path), "w") as outfile:
                data = {}
                data[dt_string] = json_info
                json_object = json.dumps(data, indent=4)
                outfile.write(json_object)
    else:
        FROM_EXP_DIR = "logs-extrinsics-mar2/optimizer-fix/exp1_v_large_z_45fov/"
        print("Note: Loading from Previous experiments env variable {}".format(FROM_EXP_DIR))
        config = load_env_config_dict(FROM_EXP_DIR)
        config.use_random = USE_RANDOM
        print("Using Config: ", config)
        path = Path('{}/exp_info.json'.format(log_dir))
        now = datetime.now()
        print("Starting Experiment At=", now)
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if not path.is_file():
            json_info = config.toDict()
            os.makedirs(log_dir, exist_ok = True)
            # path.makedirs(parents=True, exist_ok=True)
            with open(str(path), "w") as outfile:
                data = {}
                data[dt_string] = json_info
                json_object = json.dumps(data, indent=4)
                outfile.write(json_object)
    # ------------ SAVE INFO ABOUT EXPERIMENT ------------

    # mono only 
    env = SubprocVecEnv([make_env(i, run_id, env_str=ENV_STR, res=RES, 
                                    config=config,
                                    perception_net=perception_net,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    new_scene_every_episode=NEW_SCENE_EVERY_EPISODE) for i in range(num_cpu)])

    env = VecMonitor(env, log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=CHECK_FREQ, log_dir=log_dir, verbose=2)

    policy_kwargs = dict(
        features_extractor_class=PolicyFeatureExtractorExtrinsics,
        features_extractor_kwargs=dict(image_size = 128, vit=vit))

    print("Instantiating PPO model...")
    # import pdb; pdb.set_trace()

    if PPO_PATH:
        # ------------ LOAD PPO MODEL FROM CHECKPOINT ------------
        print("Loading PPO from path: {}".format(PPO_PATH))
        model = PPO.load(PPO_PATH, env=env, print_system_info=True)
        # ------------ LOAD PPO MODEL FROM CHECKPOINT ------------
    else:
        model = PPO("MultiInputPolicy",
                env,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                policy_kwargs=policy_kwargs,
                verbose=2)

    timesteps = 200000 # 1e5
    start = time()

    print("Training now...")
    model.learn(total_timesteps=int(timesteps), callback=callback)

    SAVE_PATH = "{}/{}".format(log_dir, "final_rl.pt")
    model.save(SAVE_PATH)

    save_dict = {
        'perception_net': perception_net.state_dict(),
        "vit": vit.state_dict(),
        'same_backone': False,
    }
    
    torch.save(save_dict, "{}/final_{}".format(log_dir, "perception.ckpt"))

    model_name = 'triangulation_env_ppo_runid_{}'.format(run_id)

    del model # remove to demonstrate saving and loading

    print("Training Complete, took {} hours".format((time()-start)/(60*24.)))

    model = PPO.load(SAVE_PATH)
    print("Testing the model...")
    obs = env.reset()
    step = 0 
    while True:
        print("step --> {}".format(step))
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        vis_img = env.render()
        step += 1


    print("_________Run Over________________")
