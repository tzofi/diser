from copy import copy
import cv2
import gym
from gym import spaces
import json
import numpy as np
from nuscenes.utils.data_classes import Box
import os
from pyquaternion import Quaternion
import random
from time import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import traceback
import yaml

from cross_view_transformers import cross_view_transformer
from cross_view_transformers.cross_view_transformer.losses import BinarySegmentationLoss
from models import BEVFeatureExtractor, BEVModel, dx, bx, nx
from rl_utils import SaveOnBestTrainingRewardCallback
from src.sim_nuscenes import CarlaClient
from src.utils import xz_to_xyz
from stablebaselines3.common.policies import ActorCriticPolicy 
from stablebaselines3 import PPO
from stablebaselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stablebaselines3.common.monitor import Monitor

"""
Documentation:
    gt_1/gt_2 and bboxes_1/bboxes_2 refer to (1) gt for computing reward, and (2) gt for computing loss to update perception net
    the reason it's different is reward should take into account all vehicles, but loss should only be on vehicles in FoV of cameras

"""

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_env(rank,
             perception,
             update,
             scale,
             penalty,
             bs,
             num_npcs,
             viz=False,
             optimizer=None,
             scheduler=None,
             episode_len=6,
             seed=0,
             single_cam=False,
             port=2000,
             tm_port=8000,
             mesh=False,
             rescale_reward=False,
             reward_max=0.5):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = CARLAEnv(rank,
                       port=port,
                       tm_port=tm_port,
                       perception=perception,
                       update=update,
                       scale=scale,
                       penalty=penalty,
                       bs=bs,
                       num_npcs=num_npcs,
                       viz=viz,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       episode_len=episode_len,
                       single_cam=single_cam,
                       mesh=mesh,
                       rescale_reward=rescale_reward,
                       reward_max=reward_max)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

class CARLAEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    """
    What does this environment do?

    1. For every step, we choose whether or not to place a camera and where to place it
    2. Then we collect images on all cameras we've placed
    3. We run inference on the scene and compute IoU
    4. IoU serves as reward
    5. End episode after self.rollout_length steps

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rank,
                 port=2000,
                 tm_port=8000,
                 perception=None,
                 update=True,
                 scale=1,
                 penalty=0,
                 bs=1,
                 num_npcs=15,
                 viz=False,
                 optimizer=None,
                 scheduler=None,
                 episode_len=6,
                 single_cam=False,
                 mesh=False,
                 rescale_reward=False,
                 reward_max=0.5,
                 direct_cameras=False):
        """
        :rank: the index of the env if running with multiple cpus, otherwise set to 0
        :perception: perception network
        :update: boolean for whether or not to train/finetune perception network with RL
        :scale: scale the reward by this value by multiplication
        :penalty: penalty each time a camera is added, this value is subtracted from reward
        :bs: batch size , e.g. number of images collected from CARLA at each step
        :num_npcs: number of non player characters, i.e. other vehicles, rendered in CARLA
        :viz: boolean for whether to save images for visualizing rig and predictions
        :optimizer: optimizer for training/finetuning perception network
        :episode_len: length of the episode before resetting
        :single_cam: boolean specifying whether it's a single camera env, which changes formulation slightly
        """
        super(CARLAEnv, self).__init__()

        # Initialization
        self.cameras = []
        self.steps = 0
        self.obs_fxn = self._get_obs
        self.perception = None
        self.device = torch.device("cuda:0")
        self.update = update
        if perception:
            self.perception = perception 
            self.network_params = perception.network.state_dict()
        if update:
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.loss = BinarySegmentationLoss()
            #self.loss = nn.BCEWithLogitsLoss()
            #self.loss = torch.nn.BCEWithLogitsLoss()
            #MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})

        # Parameters
        self.rollout_length = episode_len
        self.max_images = self.rollout_length
        self.cam_thresh = 0.5
        self.scale = scale
        self.penalty = penalty
        self.viz = viz
        self.single_cam = single_cam
        self.mesh = mesh
        self.rescale_reward = rescale_reward
        self.reward_max = reward_max
        self.direct_cameras = direct_cameras
        self.buffer = {"obs":[], "gt":[]}

        """
        action space: camera placement confidence, x, y, pitch, yaw

        we're going to push a padding trick for dynamic input: pad to be as large as possible
        and indicate how many inputs there are with a discrete value

        checked the bounds for x, y: x=2.090605, y=0.997059, z=0.692648

        We will normalize between -1 to 1 and then rescale
        """
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32) 

        # Ranges for rescaling actions
        self.cam_thresh_range = [0.0,1.0]
        self.x_range = [-2.090605,2.090605]
        #self.x_range = [0.02,1.7]
        self.y_range = [-0.997059,0.997059]
        #self.y_range = [-0.49,0.49]
        self.z_range = [2*0.692648,2*0.692648 + 0.5]
        #self.z_range = [1.48,1.58]
        self.pitch_range = [-20.0,20.0]
        self.yaw_range = [-180.0,180]
        self.fov_range = [50,120]
        self.ranges = [self.cam_thresh_range,
                       self.x_range,
                       self.y_range,
                       self.z_range,
                       self.pitch_range,
                       self.yaw_range,
                       self.fov_range]

        # For single cam, we only choose whether to place and height, pitch, and FoV 
        if self.single_cam:
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]),
                                           high=np.array([1.0, 1.0, 1.0, 1.0]),
                                           dtype=np.float32) 
            self.z_range = [1.49491291905-0.5, 1.49491291905+0.5]
            self.pitch_range = [-20.0,20.0]
            self.fov_range = [50,120]
            self.ranges = [self.cam_thresh_range, self.z_range, self.pitch_range, self.fov_range]
            self.def_x = 1.72200568478
            self.def_y = -0.00475453292289
            self.def_yaw = 0.0

        # For mesh, we don't predict z
        if self.mesh:
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                           high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                           dtype=np.float32) 
            self.ranges = [self.cam_thresh_range,
                           self.x_range,
                           self.y_range,
                           self.pitch_range,
                           self.yaw_range,
                           self.fov_range]

        self.bs = bs
        self.image_shape = (self.bs,self.max_images,3,224,400)
        self.intrin_shape = (self.bs,self.max_images,3,3)
        self.extrin_shape = (self.bs,self.max_images,4,4)
        self.observation_space = spaces.Dict(
            spaces={
                "image": spaces.Box(0, 255, self.image_shape, dtype=np.uint8),
                "intrinsics": spaces.Box(-np.inf, np.inf, self.intrin_shape, dtype=np.float64),
                "extrinsics": spaces.Box(-np.inf, np.inf, self.extrin_shape, dtype=np.float64),
                "cameras": spaces.Box(0,self.max_images, (1,), dtype=np.uint8),
            }
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        # Create CARLA client w/ server connection
        port = port + rank * 2
        tm_port = tm_port + 100 * rank
        self.carla = CarlaClient(port=port, tm_port=tm_port, bs=bs, num_npcs=num_npcs)

    def step(self, u):
        # this first part is just for our evaluation
        if self.direct_cameras:
            self.cameras = u
            obs, gt_1, gt_2 = self.obs_fxn()
            pred_bev, reward, loss, processed_obs = self._run_perception(copy(obs), gt_1, gt_2)
            return reward

        u = self._rescale(u)
        cam_conf = u[0]
        reward = 0
        if self.single_cam:
            u = [cam_conf, self.def_x, self.def_y, u[1], u[2], self.def_yaw, u[3]]
        elif self.mesh:
            _, z, _ = xz_to_xyz(u[1], u[2], 1.48832, False)
            u = [u[0], u[1], u[2], float(z), u[3], u[4], u[5]]

        if cam_conf > self.cam_thresh:
            self.cameras.append(u[1:])
            if self.single_cam:
                self.cameras = [u[1:]]

        self.steps += 1
        obs, gt_1, gt_2 = self.obs_fxn()
        pred_bev, reward, loss, processed_obs = self._run_perception(copy(obs), gt_1, gt_2)
        if self.update:
            loss = self._train(copy(obs), gt_1)
        
        if self.rescale_reward:
            e_min = 0
            e_max = self.reward_max
            reward = (2*(np.minimum(reward, e_max) - e_min) / (e_max - e_min) - 1) # scale it to [-1,1]

        if cam_conf > self.cam_thresh:
            reward -= self.penalty

        if self.viz:
            self.perception.viz(processed_obs, gt_1, pred_bev, self.steps)

        done = True if self.steps % self.rollout_length == 0 else False

        """ logging """
        print("Reward: {}, Loss: {}, Cam placement: {}, Cameras: {}, X: {}, Y: {}, Z: {}, Pitch: {}, Yaw: {}, FoV: {}".format(reward, loss, cam_conf, len(self.cameras), u[1], u[2], u[3], u[4], u[5], u[6]))

        # cleaning up memory
        del processed_obs
        del loss
        return obs, reward, done, {}
        
    def reset(self):
        """ logging """
        print("*** Resetting ***")
        #self.carla.update_scene()
        self.steps = 0
        self.cameras = []
        obs, _, _ = self._get_obs()
        return obs
        
    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        """
        returns: B x N x C x H x W
        """

        if len(self.cameras):
            # Assuming one session (thus only indexing 0)
            extrins = self._get_poses(self.cameras)
            session = self.carla.rollout(self.cameras)[0]
            imgs = session[0]
            intrins = session[1]
            
            #_, intrins, extrins = self._get_info_offline(self.cameras)

            boxes_1 = session[2]
            boxes_2 = session[3]
            bev_1 = self._all_boxes_to_bev(boxes_1)
            bev_2 = self._all_boxes_to_bev(boxes_2)

            # Logic for padding dynamic input
            # We later will remove padding -- this gets us past stablebaselines size checks
            if imgs.shape[1] < self.max_images:
                diff = self.max_images - imgs.shape[1]
                bs = imgs.shape[0]
                #image = np.repeat(np.expand_dims(np.random.rand(diff, *self.image_shape[1:]),0),bs,0)
                #i = np.repeat(np.expand_dims(np.eye(self.intrin_shape[1]),0),diff,0)
                #e = np.repeat(np.expand_dims(np.eye(self.extrin_shape[1]),0),diff,0)

                image = np.repeat(np.expand_dims(np.random.rand(diff, *self.image_shape[2:]),0),self.bs,0)
                i = np.repeat(np.expand_dims(np.eye(self.intrin_shape[2]),0),diff,0)
                e = np.repeat(np.expand_dims(np.eye(self.extrin_shape[2]),0),diff,0)

                #print(imgs.shape, image.shape)
                imgs = np.concatenate((imgs,image),axis=1)
                intrins = np.concatenate((intrins,i),axis=0)
                extrins = np.concatenate((extrins,e),axis=0)

            extrins = np.repeat(np.expand_dims(extrins,0),self.bs,0)
            intrins = np.repeat(np.expand_dims(intrins,0),self.bs,0)

            cameras = len(self.cameras)
        else:
            #imgs = np.expand_dims(np.zeros(self.image_shape),0)
            imgs = np.zeros(self.image_shape)
            #bev_1 = [np.zeros((1,200,200))]
            #bev_2 = [np.zeros((1,200,200))]
            bev_1 = np.repeat(np.expand_dims(np.zeros((1,200,200)),0),self.bs,0)
            bev_2 = np.repeat(np.expand_dims(np.zeros((1,200,200)),0),self.bs,0)
            #intrins = np.expand_dims(np.repeat(np.expand_dims(np.eye(self.intrin_shape[1]),0),self.max_images,0),0)
            #extrins = np.expand_dims(np.repeat(np.expand_dims(np.eye(self.extrin_shape[1]),0),self.max_images,0),0)
            intrins = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.eye(self.intrin_shape[2]),0),self.max_images,0),0),self.bs,0)
            extrins = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.eye(self.extrin_shape[2]),0),self.max_images,0),0),self.bs,0)
            cameras = 1

        #print(imgs.shape)
        #print(intrins.shape)
        #print(extrins.shape)
        #print(cameras)

        cameras = np.expand_dims(np.array(cameras),0)
        obs = {
            "image": imgs,
            "intrinsics": intrins,
            "extrinsics": extrins,
            "cameras": cameras,
        }

        return obs, np.array(bev_1), np.array(bev_2)

    def _get_info_offline(self, cameras):
        width = 400
        height = 224
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        with open('nusccalib.json', 'r') as reader:
            calib = json.load(reader)["scene-0129"]
        info = {}
        for k, cal in calib.items():
            if k == 'LIDAR_TOP':
                continue
            trans = [cal['trans'][0], -cal['trans'][1], cal['trans'][2]]
            intrins = np.identity(3)
            intrins[0, 2] = width / 2.0
            intrins[1, 2] = height / 2.0
            intrins[0, 0] = intrins[1, 1] = width / (2.0 * np.tan(cal['fov'] * np.pi / 360.0))
            coordmat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            rot = np.matmul(
                Quaternion(axis=[0, 0, 1], angle=np.radians(-(cal['yaw']))).rotation_matrix,
                #Quaternion(axis=[0, 1, 0], angle=np.radians(-(-10))).rotation_matrix,
                np.linalg.inv(coordmat))
            quat = Quaternion(matrix=rot)
            rc = quat.rotation_matrix @ np.array(trans)
            extrins = np.random.rand(4,4)
            extrins[:3,:3] = quat.rotation_matrix
            extrins[:3,3] = -1 * rc
            extrins[3,:] = np.array([0,0,0,1])
            info[k] = {'trans': trans, 'intrins': intrins,
                    'rot': quat, 'extrins': extrins}

        extrins = []
        intrins = []
        for idx in range(len(cameras)):
            cam = cams[idx]
            extrin = info[cam]["extrins"]
            intrin = info[cam]["intrins"]
            extrins.append(extrin)
            intrins.append(intrin)

        return np.expand_dims(np.stack(imgs),0), np.stack(intrins), np.stack(extrins)

    def _get_poses(self, cameras):
        """
        cameras: list of x,y,pitch,yaw
        """
        matrices = []
        for cam in cameras:
            x = cam[0]
            y = cam[1]
            z = cam[2]
            pitch = cam[3]
            yaw = cam[4]

            # From unreal coordinate system to opencv
            trans = [x,-y,z]
            coordmat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            yaw = np.matmul(
                Quaternion(axis=[0, 0, 1], angle=np.radians(-yaw)).rotation_matrix,
                np.linalg.inv(coordmat))
            pitch = np.matmul(
                Quaternion(axis=[0, 1, 0], angle=np.radians(-pitch)).rotation_matrix,
                np.linalg.inv(coordmat))
            
            rot = yaw @ pitch
            quat = Quaternion(matrix=rot)
            rc = quat.rotation_matrix @ np.array(trans)
            extrins = np.random.rand(4,4)
            extrins[:3,:3] = quat.rotation_matrix
            extrins[:3,3] = -1 * rc
            extrins[3,:] = np.array([0,0,0,1])
            
            matrices.append(extrins)
        return np.stack(matrices,0)

    def _rescale(self, actions):
        """
        actions are normalized between -1 and 1, so we need to rescale
        """
        rescaled_actions = []
        min_value = -1.0
        max_value = 1.0
        for i, act in enumerate(actions):
            a = self.ranges[i][0]
            b = self.ranges[i][1]
            rescaled = (b-a) * (act - min_value) / (max_value - min_value) + a
            rescaled_actions.append(rescaled)
        return np.array(rescaled_actions)

    def _train(self, obs, gt_2):
        self.perception.train()

        self.buffer["obs"].append(obs)
        self.buffer["gt"].append(gt_2)

        BUFFER_SIZE = 6
        if len(self.buffer["obs"]) > BUFFER_SIZE:
            self.buffer["obs"].pop(0)
            self.buffer["gt"].pop(0)

        total_loss = 0.0
        EPOCHS = 2
        for i in range(EPOCHS):
            preds = []
            gts = []
            for j in range(len(self.buffer["obs"])):
                obs = self.buffer["obs"][j]
                gt = torch.tensor(self.buffer["gt"][j]).to(self.device)
                pred = self.perception(obs)
                preds.append(pred["bev"])
                gts.append(gt)

            gt_batch = torch.cat(gts, 0)
            pred_batch = torch.cat(preds, 0)
            self.optimizer.zero_grad()
            loss = self.loss(pred, {"bev": gt})
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        total_loss = total_loss / EPOCHS
        return total_loss

    def _run_perception(self, obs, gt_1, gt_2):
        """
        Function to compute the reward (IoU) and update perception network when self.update is True

        :obs: dictionary containing images, extrinsics, intrinsics, and cam indices
        :gt_1: bev map containing ALL vehicles in the scene --> we use this for the reward
        :gt_2: bev map containing ONLY vehicles in the FoV of the cameras --> we use this to compute the loss

        :return: predicted bev map, reward (self.scale x IoU), loss, and processed obs (i.e. padding removed)
        """
        with torch.no_grad():
            pred = self.perception(obs)

        #print(pred["bev"].shape)
        #print(gt_1.shape)
        
        loss = 0
        """
        if self.update and len(self.cameras) > 0:
            gt_2 = torch.tensor(gt_2).to(self.device)
            #print(torch.max(gt_2), torch.min(gt_2))
            #print(torch.max(pred['bev']))
            loss = self.loss(pred, {"bev": gt_2})
            #loss = self.loss(pred['bev'], gt_2)
            loss.backward()
            loss = loss.item()
            gt_2 = gt_2.detach().cpu().numpy()

            # Accumulated Gradient Approach
            if self.steps % 120 == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.network_params = self.perception.network.state_dict()
        """

        reward = self._compute_reward(copy(pred["bev"]), gt_1)
        reward *= self.scale
        return pred["bev"], reward, loss, obs

    def _compute_reward(self, pred, gt):
        """
        compute iou
        """
        thresholds=torch.FloatTensor([0.4])
        label = torch.tensor(gt).to(self.device)
        pred = pred.detach().cpu().sigmoid().reshape(-1)
        label = label.detach().cpu().bool().reshape(-1)
        pred = pred[:, None] >= thresholds[None]
        label = label[:, None]
        tp = (pred & label).sum(0)
        fp = (pred & ~label).sum(0)
        fn = (~pred & label).sum(0)
        ious = tp / (tp + fp + fn + 1e-7)
        return float(ious[0])

    def _all_boxes_to_bev(self, boxes):
        """
        returns a list of bev maps -- one per sample
        """
        bevs = []
        for box in boxes:
            bev, _ = self._boxes_to_bev(np.array(box))
            bevs.append(bev)
        return bevs

    def _boxes_to_bev(self, gt):
        """
        converts 3d cuboids into bev map
        """
        def get_box(box):
            diffw = box[:3, 1] - box[:3, 2]
            diffl = box[:3, 0] - box[:3, 1]
            diffh = box[:3, 4] - box[:3, 0]

            center = (box[:3, 4] + box[:3, 2]) / 2
            # carla flips y axis
            center[1] = -center[1]

            dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]
            if 0 in dims: return None

            # might need to be transposed? nvm
            rot = np.zeros((3, 3))
            rot[:, 1] = diffw / dims[0]
            rot[:, 0] = diffl / dims[1]
            rot[:, 2] = diffh / dims[2]

            quat = Quaternion(matrix=rot)
            # again, carla flips y axis
            newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)

            nbox = Box(center, dims, newquat)
            return nbox

        boxes = [get_box(box) for box in gt]

        img = np.zeros((int(nx[0]), int(nx[1])))
        for nbox in boxes:
            if nbox is None: continue
            pts = nbox.bottom_corners()[:2].T
            pts = np.round(
                (pts - bx[:2] + dx[:2] / 2.) / dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return np.expand_dims(img,0), boxes

if __name__ == "__main__":
    """ pass in log dir path as arg or path to yaml config """
    TEST_MODE = 2
    log_dir = "./{}".format(sys.argv[1])
    ckpt_path = "model.ckpt"
    UPDATE = True
    SCALE = 1
    PENALTY = 0 #0.005
    VIZ = False
    NUM_CPU = 1
    BATCH_SIZE = 1
    num_npcs = 15
    optimizer = None
    scheduler = None
    CKPT_FREQ = 64
    LOAD_CKPT = None
    NUM_STEPS = 10000
    EPISODE_LEN = 6
    SINGLE_CAM = False
    PORT = 2000
    TM_PORT = 8000
    MESH = False
    RESCALE_REWARD = False
    REWARD_MAX = 0.5
    LR = 4e-5
    if sys.argv[1][-4:] == "yaml":
        with open(sys.argv[1], 'r') as fp:
            args = yaml.safe_load(fp)
        TEST_MODE = args["TEST_MODE"]
        log_dir = args["log_dir"]
        ckpt_path = args["ckpt_path"]
        UPDATE = args["UPDATE"]
        SCALE = args["SCALE"]
        PENALTY = args["PENALTY"]
        VIZ = args["VIZ"]
        NUM_CPU = args["NUM_CPU"]
        BATCH_SIZE = args["BATCH_SIZE"]
        num_npcs = args["NPCS"]
        CKPT_FREQ = args["CKPT_FREQ"]
        LOAD_CKPT = args["LOAD_CKPT"]
        NUM_STEPS = args["NUM_STEPS"]
        EPISODE_LEN = args["EPISODE_LEN"]
        SINGLE_CAM = args["SINGLE_CAM"]
        PORT = args["PORT"]
        TM_PORT = args["TM_PORT"]
        MESH = args["MESH"]
        RESCALE_REWARD = args["RESCALE_REWARD"]
        REWARD_MAX = args["REWARD_MAX"]

        if "LR" in args:
            LR = float(args["LR"])
        print(args)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)



    if TEST_MODE == 3:

        done = False
        attempt = 0
        original_ckpt_path = ckpt_path
        while not done:
            try:
                ckpt = torch.load(ckpt_path)
                hparams = ckpt['hyper_parameters']
                perception_net = BEVModel(ckpt_path, torch.device("cuda:0"))
                callback = None
                if UPDATE:
                    for param in perception_net.parameters():
                        param.requires_grad = True
                    parameters = [x for x in perception_net.parameters() if x.requires_grad]
                    optimizer = torch.optim.AdamW(parameters, lr=4e-7, weight_decay=1e-10)
                    optimizer.load_state_dict(ckpt['optimizer_states'][0])
                    for g in optimizer.param_groups:
                        g['lr'] = LR

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, div_factor=10, pct_start=0.3, final_div_factor=10, cycle_momentum=False, max_lr=LR, total_steps=1000000)
                    scheduler.load_state_dict(ckpt['lr_schedulers'][0])
                    scheduler.total_steps = 100000
                    print("Loaded state dict!")
                    callback = SaveOnBestTrainingRewardCallback(check_freq=CKPT_FREQ, log_dir=log_dir, update=True, hparams=hparams, verbose=2)
                else:
                    callback = SaveOnBestTrainingRewardCallback(check_freq=CKPT_FREQ, log_dir=log_dir, update=False, verbose=2)


                for name, param in perception_net.network.named_parameters():
                    print("Sanity check - Name: {}, Norm: {}".format(name, torch.norm(param)))
                    break

                num_cpu = NUM_CPU  # Number of processes to use
                env = SubprocVecEnv([make_env(i,
                                              perception_net,
                                              UPDATE,
                                              SCALE,
                                              PENALTY,
                                              BATCH_SIZE,
                                              num_npcs,
                                              viz=VIZ,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              episode_len=EPISODE_LEN,
                                              single_cam=SINGLE_CAM,
                                              port=PORT,
                                              tm_port=TM_PORT,
                                              mesh=MESH,
                                              rescale_reward=RESCALE_REWARD,
                                              reward_max=REWARD_MAX) for i in range(num_cpu)])
                env = VecMonitor(env, log_dir)
                policy_kwargs = dict(
                    features_extractor_class=BEVFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=2048,ckpt_path=original_ckpt_path),
                )
                model = PPO("MultiInputPolicy",
                            env,
                            n_steps=2,
                            batch_size=2,
                            #n_steps=18,
                            #batch_size=6,
                            policy_kwargs=policy_kwargs,
                            verbose=2)
                if LOAD_CKPT:
                    print("LOADED CKPT FROM {}".format(LOAD_CKPT))
                    model = model.load("{}/best_model".format(LOAD_CKPT), env=env)
                start = time()

                attempt = 0 
                done = False
                model.learn(total_timesteps=NUM_STEPS, callback=callback)
                model.save(log_dir)
                print("Total time: {}".format(time() - start))
                done = True
            except Exception as e:
                print("RESTARTING")
                print(traceback.format_exc())
                attempt += 1
                os.system("mv {}/monitor.csv {}/{}_monitor.csv".format(log_dir, log_dir, attempt))
                TM_PORT += 11
                LOAD_CKPT = log_dir
                if UPDATE:
                    ckpt_path = "{}/best_perception_model/perception_0.ckpt".format(log_dir)
                    print("New perception ckpt: {}".format(ckpt_path))
                try:
                    del env
                    del policy_kwargs
                    del model
                except:
                    pass
                continue
        print("Done")
        exit()

    ckpt = torch.load(ckpt_path)
    hparams = ckpt['hyper_parameters']
    perception_net = BEVModel(ckpt_path, torch.device("cuda:0"))
    callback = None
    if UPDATE:
        for param in perception_net.parameters():
            param.requires_grad = True
        parameters = [x for x in perception_net.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=4e-7, weight_decay=1e-10)
        optimizer.load_state_dict(ckpt['optimizer_states'][0])
        for g in optimizer.param_groups:
            g['lr'] = LR

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, div_factor=10, pct_start=0.3, final_div_factor=10, cycle_momentum=False, max_lr=LR, total_steps=1000000)
        scheduler.load_state_dict(ckpt['lr_schedulers'][0])
        scheduler.total_steps = 100000
        print("Loaded state dict!")
        callback = SaveOnBestTrainingRewardCallback(check_freq=CKPT_FREQ, log_dir=log_dir, update=True, hparams=hparams, verbose=2)
    else:
        callback = SaveOnBestTrainingRewardCallback(check_freq=CKPT_FREQ, log_dir=log_dir, update=False, verbose=2)

    if TEST_MODE == 2:
        """ Phase 2 testing """
        env = CARLAEnv(0, perception_net, UPDATE, SCALE, PENALTY, BATCH_SIZE, num_npcs, viz=VIZ, optimizer=optimizer, episode_len=EPISODE_LEN, single_cam=SINGLE_CAM, port=PORT, tm_port=TM_PORT, mesh=MESH)
        policy_kwargs = dict(
            features_extractor_class=BEVFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=2048,ckpt_path=ckpt_path),
        )
        model = PPO("MultiInputPolicy",
                    env,
                    n_steps=18,
                    batch_size=6,
                    policy_kwargs=policy_kwargs,
                    verbose=1)
        model.learn(10000)

    if TEST_MODE == 1:
        """ Phase 1 Testing """
        ckpt_path="model.ckpt"
        env = CARLAEnv(0, perception_net) 
        for i in range(12):
            u = np.random.rand(6,)
            imgs, reward, done, _ = env.step(u)
            if done:
                env.reset()

    if TEST_MODE == 0:
        """ Testing trained model """
        env = CARLAEnv(0, perception=perception_net, update=UPDATE, scale=SCALE, penalty=PENALTY, bs=BATCH_SIZE, num_npcs=num_npcs, viz=VIZ, optimizer=optimizer, episode_len=EPISODE_LEN, single_cam=SINGLE_CAM, port=PORT, tm_port=TM_PORT, mesh=MESH) 
        policy_kwargs = dict(
            features_extractor_class=BEVFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=2048,ckpt_path=ckpt_path),
        )
        model = PPO("MultiInputPolicy",
                    env,
                    n_steps=18,
                    batch_size=6,
                    policy_kwargs=policy_kwargs,
                    verbose=2)
        model = model.load("{}/best_model".format(LOAD_CKPT), env=env)
        print("LOADED CKPT FROM {}".format(LOAD_CKPT))
        obs = env.reset()
        for i in range(12):
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                env.reset()

    if TEST_MODE == "DEEP_TEST":
        """ Testing trained model """
        env = CARLAEnv(0,
                perception=perception_net,
                update=UPDATE,
                scale=SCALE,
                penalty=PENALTY,
                bs=BATCH_SIZE,
                num_npcs=num_npcs,
                viz=VIZ,
                optimizer=optimizer,
                episode_len=EPISODE_LEN,
                single_cam=SINGLE_CAM,
                port=PORT,
                tm_port=TM_PORT,
                mesh=MESH)
        policy_kwargs = dict(
            features_extractor_class=BEVFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=2048,ckpt_path=ckpt_path),
        )
        model = PPO("MultiInputPolicy",
                    env,
                    n_steps=18,
                    batch_size=6,
                    policy_kwargs=policy_kwargs,
                    verbose=2)
        model = model.load("{}/best_model".format(LOAD_CKPT), env=env)
        print("LOADED CKPT FROM {}".format(LOAD_CKPT))
        obs = env.reset()
        
        cam_confs = []
        xs = []
        ys = []
        zs = []
        pitchs = []
        yaws = []
        fovs = []
        rewards = []

        cam_conf = []
        x = []
        y = []
        z = []
        pitch = []
        yaw = []
        fov = []
        reward = []

        episode_idx = 0
        for i in range(NUM_STEPS):
            action, _states = model.predict(obs)

            u = env._rescale(action)
            cam_conf.append(u[0])
            x.append(u[1])
            y.append(u[2])
            z.append(u[3])
            pitch.append(u[4])
            yaw.append(u[5])
            fov.append(u[6])

            obs, r, done, _ = env.step(action)
            reward.append(r)
            if done:
                episode_idx += 1
                env.reset()

                cam_confs.append(np.array(sorted(cam_conf)))
                xs.append(np.array(sorted(x)))
                ys.append(np.array(sorted(y)))
                zs.append(np.array(sorted(z)))
                pitchs.append(np.array(sorted(pitch)))
                yaws.append(np.array(sorted(yaw)))
                fovs.append(np.array(sorted(fov)))
                rewards.append(reward)

                cam_conf = []
                x = []
                y = []
                z = []
                pitch = []
                yaw = []
                fov = []
                reward = []

        print("CAM CONFS - Mean: {}, Std: {}".format(np.mean(cam_confs, axis=0), np.std(cam_confs, axis=0)))
        print("X - Mean: {}, Std: {}".format(np.mean(xs, axis=0), np.std(xs, axis=0)))
        print("Y - Mean: {}, Std: {}".format(np.mean(ys, axis=0), np.std(ys, axis=0)))
        print("Z - Mean: {}, Std: {}".format(np.mean(zs, axis=0), np.std(zs, axis=0)))
        print("PITCH - Mean: {}, Std: {}".format(np.mean(pitchs, axis=0), np.std(pitchs, axis=0)))
        print("YAW - Mean: {}, Std: {}".format(np.mean(yaws, axis=0), np.std(yaws, axis=0)))
        print("FOV - Mean: {}, Std: {}".format(np.mean(fovs, axis=0), np.std(fovs, axis=0)))
        print("REWARD - Mean: {}, Std: {}".format(np.mean(rewards, axis=0), np.std(rewards, axis=0)))

    if TEST_MODE == "DEEP_TEST_1CAM":
        """ Testing trained model """
        env = CARLAEnv(0,
                perception=perception_net,
                update=UPDATE,
                scale=SCALE,
                penalty=PENALTY,
                bs=BATCH_SIZE,
                num_npcs=num_npcs,
                viz=VIZ,
                optimizer=optimizer,
                episode_len=EPISODE_LEN,
                single_cam=SINGLE_CAM,
                port=PORT,
                tm_port=TM_PORT)
        policy_kwargs = dict(
            features_extractor_class=BEVFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=2048,ckpt_path=ckpt_path),
        )
        model = PPO("MultiInputPolicy",
                    env,
                    n_steps=18,
                    batch_size=6,
                    policy_kwargs=policy_kwargs,
                    verbose=2)
        model = model.load("{}/best_model".format(LOAD_CKPT), env=env)
        print("LOADED CKPT FROM {}".format(LOAD_CKPT))
        obs = env.reset()
        
        cam_confs = []
        zs = []
        pitchs = []
        fovs = []
        rewards = []

        cam_conf = []
        z = []
        pitch = []
        fov = []
        reward = []

        episode_idx = 0
        for i in range(NUM_STEPS):
            action, _states = model.predict(obs)

            u = env._rescale(action)
            cam_conf.append(u[0])
            z.append(u[1])
            pitch.append(u[2])
            fov.append(u[3])

            obs, r, done, _ = env.step(action)
            reward.append(r)
            if done:
                episode_idx += 1
                env.reset()

                cam_confs.append(np.array(sorted(cam_conf)))
                zs.append(np.array(sorted(z)))
                pitchs.append(np.array(sorted(pitch)))
                fovs.append(np.array(sorted(fov)))
                rewards.append(reward)

                cam_conf = []
                z = []
                pitch = []
                fov = []
                reward = []

        print("CAM CONFS - Mean: {}, Std: {}".format(np.mean(cam_confs, axis=0), np.std(cam_confs, axis=0)))
        print("Z - Mean: {}, Std: {}".format(np.mean(zs, axis=0), np.std(zs, axis=0)))
        print("PITCH - Mean: {}, Std: {}".format(np.mean(pitchs, axis=0), np.std(pitchs, axis=0)))
        print("FOV - Mean: {}, Std: {}".format(np.mean(fovs, axis=0), np.std(fovs, axis=0)))
        print("REWARD - Mean: {}, Std: {}".format(np.mean(rewards, axis=0), np.std(rewards, axis=0)))

    if TEST_MODE == "BASELINE":
        """ This implements random search as a baseline """
        env = CARLAEnv(0, perception=perception_net, update=UPDATE, scale=SCALE, penalty=PENALTY, bs=BATCH_SIZE, num_npcs=num_npcs, viz=VIZ, optimizer=optimizer, episode_len=EPISODE_LEN, single_cam=SINGLE_CAM, port=PORT, tm_port=TM_PORT,
                mesh=MESH,
                rescale_reward=RESCALE_REWARD,
                reward_max=REWARD_MAX) 
        obs = env.reset()

        start_time = time()
        monitor = open(os.path.join(log_dir, "monitor.csv"),"w")
        monitor.write("#{{\"t_start\": {}, \"env_id\": null}}\n".format(start_time))
        monitor.write("r,l,t\n")
        monitor.flush()

        reward_sum = 0
        for i in range(NUM_STEPS):
            # order: yes/no, x, y, z, pitch, yaw, fov
            action = np.random.uniform(low=-1.0, high=1.0, size=7)
            if SINGLE_CAM:
                action = np.random.uniform(low=-1.0, high=1.0, size=4)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                total_time = time() - start_time
                monitor.write("{},{},{}\n".format(reward_sum,EPISODE_LEN,total_time))
                monitor.flush()
                reward_sum = 0
                env.reset()

        monitor.close()

    if TEST_MODE == "EVAL_ACROSS":
        env = CARLAEnv(0, perception=perception_net, update=UPDATE, scale=SCALE, penalty=PENALTY, bs=BATCH_SIZE, num_npcs=num_npcs, viz=VIZ, optimizer=optimizer, episode_len=EPISODE_LEN, single_cam=SINGLE_CAM, port=PORT, tm_port=TM_PORT,
                mesh=MESH,
                rescale_reward=RESCALE_REWARD,
                reward_max=REWARD_MAX,
                direct_cameras=True) 
        obs = env.reset()

        candidates = []
        log = open(sys.argv[2],"r")
        lines = []
        for i, line in enumerate(log):
            lines.append(line.replace("\n",""))
        log.close()
        cameras = []
        sum_of_rewards = 0
        for i, line in enumerate(lines):
            if "Reward" in line: # and "Resetting" in lines[i + 1]:
                split = line.split(",")
                candidate = {}
                for s in split:
                    s = s.replace(" ","").split(":")
                    candidate[s[0].lower()] = float(s[1]) 

                sum_of_rewards += candidate['reward']
                #print(sum_of_rewards, candidate['reward'])
                u = [candidate["x"], candidate["y"], candidate["z"], candidate["pitch"], candidate["yaw"], candidate["fov"]]
                cameras.append(u)

            if i != len(lines) - 1 and "Resetting" in lines[i+1]:
                cur_min = -1 * np.inf
                if len(candidates) > 0:
                    cur_min = min([c['reward'] for c in candidates])
                #print(sum_of_rewards, cur_min, len(candidates))
                if len(candidates) < 20:
                    candidates.append({"reward": sum_of_rewards, "rig": cameras})
                elif sum_of_rewards > cur_min:
                    candidates = [c for c in candidates if c['reward'] > cur_min]
                    candidates.append({"reward": sum_of_rewards, "rig": cameras})
                    #print([c["reward"] for c in candidates])

                sum_of_rewards = 0
                cameras = []

        """
        print(candidates)
        print(len(candidates))
        for c in candidates:
            print(len(c['rig']))
            print(c["reward"])
        exit()
        """
        print(candidates)

        start_time = time()
        monitor = open(os.path.join(log_dir, "eval.csv"),"w")
        monitor.flush()

        reward_sum = 0
        eval_per_rig = 25
        for i in range(len(candidates)):
            cameras = candidates[i]["rig"]
            for j in range(eval_per_rig):
                reward = env.step(cameras)
                reward_sum += reward
                env.reset()
            total_time = time() - start_time
            monitor.write("{},{}\n".format(reward_sum,cameras))
            monitor.flush()
            reward_sum = 0
            env.reset()

        monitor.close()
