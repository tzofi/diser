import torch
import torchvision
import os
import numpy as np
import json
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
from pathlib import Path

from .augmentations import StrongAug, GeometricAug
from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, choose_cams, parse_calibration, sample_augmentation
from .transforms import Sample
from .common import get_view_matrix

CARLA_CAMORDER = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_BACK_RIGHT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_FRONT_LEFT': 5,
}
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

#'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
#         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],

class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class CarlaBEV(object):
    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf, ret_boxes, pitch_adjust=0, limit=None, noadjust=False):
        self.dataroot = dataroot
        self.is_train = is_train
        self.noadjust=noadjust
        self.data_aug_conf = data_aug_conf
        self.ret_boxes = ret_boxes
        self.grid_conf = grid_conf
        self.pitch_adjust = pitch_adjust
        self.ixes = self.get_ixes()
        if limit:
            self.ixes = self.ixes[:limit]

        # hard code this for now
        with open('nusccalib.json', 'r') as reader:
            self.nusccalib = json.load(reader)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print('Carla sim:', len(self), 'is train:', self.is_train)

    def get_ixes(self):
        fs = os.listdir(self.dataroot)
        fs = [f for f in fs if os.path.isdir(os.path.join(self.dataroot, f))]
        timesteps = []
        for f in fs:
            imgix = set(
                [int(fo.split('_')[0]) for fo in os.listdir(os.path.join(self.dataroot, f)) if
                 fo != 'info.json' and fo[-4:] == '.jpg'])
            for img in imgix:
                timesteps.append((f, img))
        timesteps = sorted(timesteps, key=lambda x: (int(x[0].split('_')[0]), x[1]))
        return timesteps
        '''
        splitix = int(len(timesteps) * 7 / 8)
        if self.is_train:
            return timesteps[:splitix]
        else:
            return timesteps[splitix:]
        '''

    def get_image_data(self, f, fo, cams, calib, cam_adjust, use_cam_name=False, augment='none', random_angle=None):
        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)

        imgs = []
        rots = []
        trans = []
        intrins = []
        extrins = []
        cam_rig = []
        cam_channel = []
        cal = parse_calibration(calib, width=self.data_aug_conf['W'], height=self.data_aug_conf['H'],
                                cam_adjust=cam_adjust,pitch_adjust=self.pitch_adjust,noadjust=self.noadjust,random_angle=random_angle)
        path = None
        for cam in cams:
            if use_cam_name:
                path = os.path.join(f, f'{fo:04}_{cam}.jpg')
            else:
                path = os.path.join(self.dataroot, f, f'{fo:04}_{CARLA_CAMORDER[cam]:02}.jpg')
            image = Image.open(path)

            intrin = torch.Tensor(cal[cam]['intrins'])
            rot = torch.Tensor(cal[cam]['rot'].rotation_matrix)
            tran = torch.Tensor(cal[cam]['trans'])
            extrin = torch.Tensor(cal[cam]['extrins'])

            h = self.data_aug_conf['H']
            w = self.data_aug_conf['W']
            min_dim = min(h,w)
            h_resize = self.data_aug_conf['H'] #h# + top_crop
            w_resize = self.data_aug_conf['W'] #w
            #image_new = image.resize((min_dim, min_dim), resample=Image.BILINEAR)
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            #image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
            I = np.float32(intrin)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            #I[1, 2] -= top_crop
            img = self.img_transform(image_new)
            #print(img.shape)
            #img_new = torch.zeros(3,h,w)
            #print(img_new.shape)
            #print(img_new[:,:,128:-128].shape)
            #img_new[:,:,128:-128] = img
            #imgs.append(self.img_transform(image_new))
            #print([img.shape, img_new.shape])
            imgs.append(img)
            intrins.append(torch.tensor(I))

            #imgs.append(normalize_img(img))
            #intrins.append(intrin)
            extrins.append(extrin.tolist())
            rots.append(rot)
            trans.append(tran)
            cam_rig.append(CARLA_CAMORDER[cam])
            cam_channel.append(cam)

        return {
            'cam_ids': torch.LongTensor(cam_rig),
            'cam_channels': cam_channel,
            'intrinsics': torch.stack(intrins,0),
            'extrinsics': torch.tensor(np.float32(extrins)),
            'rots': rots,
            'trans': trans,
            'image': torch.stack(imgs,0),
        }

    def get_binimg(self, gt):
        img = np.zeros((self.nx[0], self.nx[1]))
        boxes = []
        for box in gt:
            diffw = box[:3, 1] - box[:3, 2]
            diffl = box[:3, 0] - box[:3, 1]
            diffh = box[:3, 4] - box[:3, 0]

            center = (box[:3, 4] + box[:3, 2]) / 2
            # carla flips y axis
            center[1] = -center[1]

            dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]

            # might need to be transposed? nvm
            rot = np.zeros((3, 3))
            rot[:, 1] = diffw / dims[0]
            rot[:, 0] = diffl / dims[1]
            rot[:, 2] = diffh / dims[2]

            quat = Quaternion(matrix=rot)
            # again, carla flips y axis
            newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)

            nbox = Box(center, dims, newquat)
            boxes.append(nbox)

            pts = nbox.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0), boxes

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        f, fo = self.ixes[index]

        cams = choose_cams(self.is_train, self.data_aug_conf)

        with open(os.path.join(self.dataroot, f, 'info.json'), 'r') as reader:
            gt = json.load(reader)

        # backwards compatability
        if not 'cam_adjust' in gt:
            gt['cam_adjust'] = {k: {'fov': 0.0, 'yaw': 0.0} for k in CARLA_CAMORDER}

        sample = self.get_image_data(f, fo, cams, self.nusccalib[gt['scene_calib']], gt['cam_adjust'])
        binimg, boxes = self.get_binimg(np.array(gt['boxes'][fo]))
        
        return sample



class CarlaBEV2(CarlaBEV):
    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf, ret_boxes, pts_occlusion_filter, pitch_adjust=0, 
            limit=None, noadjust=False, random_angle=False):
        super(CarlaBEV2, self).__init__(dataroot, is_train, data_aug_conf, grid_conf, ret_boxes, pitch_adjust, limit, noadjust)
        self.pts_occlusion_filter = pts_occlusion_filter
        self.random_angle = random_angle
        print('CARLA BEV 2 size:', len(self), '| is_train:', self.is_train, '| Occlusion filter:', pts_occlusion_filter)
        bev = {'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
        self.view = get_view_matrix(**bev)

    def get_ixes(self):
        timesteps = []
        for path in Path(self.dataroot).rglob('info.json'):
            f = str(path.parents[0])
            imgix = set(
                [int(fo.split('_')[0]) for fo in os.listdir(f) if fo != 'info.json' and fo[-4:] == '.jpg'])
            for img in imgix:
                timesteps.append((f, img))

        timesteps = sorted(timesteps, key=lambda x: (x[0], x[1]))
        return timesteps

    def get_binimg(self, gt, lidname, pts_occlusion_filter, random_angle=None):
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
        """
        if random_angle is not None and len(random_angle)==2:
            #print("Binimg: {}".format(random_angle))
            if random_angle[0] != 0:
                # pitch
                quat = Quaternion(axis=[0, 1, 0], angle=np.radians(-(random_angle[0])))
                for box in boxes:
                    box.rotate(quat)
            elif random_angle[1] != 0:
                # yaw
                quat = Quaternion(axis=[0, 0, 1], angle=np.radians(-(random_angle[1])))
                for box in boxes:
                    box.rotate(quat)
        """

        # only read in the lidar if we have to
        if pts_occlusion_filter > 0 and len(boxes) > 0:
            rboxes = np.array([box2rbox(box) for box in boxes])
            scan = read_carla_lidar(lidname)
            inside_box = points_in_rbbox(scan.T, rboxes)
            kept = inside_box.sum(axis=0) >= pts_occlusion_filter
            boxes = [box for boxi, box in enumerate(boxes) if kept[boxi]]

        img = np.zeros((int(self.nx[0]), int(self.nx[1])))
        for nbox in boxes:
            if nbox is None: continue
            pts = nbox.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0), boxes

    def __getitem__(self, index):
        f, fo = self.ixes[index]

        cams = choose_cams(self.is_train, self.data_aug_conf)

        with open(os.path.join(f, 'info.json'), 'r') as reader:
            gt = json.load(reader)
        # backwards compatability
        if not 'cam_adjust' in gt:
            gt['cam_adjust'] = {k: {'fov': 0.0, 'yaw': 0.0} for k in CARLA_CAMORDER}

        random_angle = None
        # [ pitch, yaw ]
        if self.random_angle:
            extr = np.random.randint(0,2)
            degree = np.random.randint(-20,21)
            if extr == 0:
                random_angle = [degree,0]
            else:
                random_angle = [0,degree]

        sample = self.get_image_data(f, fo, cams, self.nusccalib[gt['scene_calib']], gt['cam_adjust'], use_cam_name=False, random_angle=random_angle)
        binimg, boxes = self.get_binimg(np.array(gt['boxes'][fo]),
                                        os.path.join(f, f'{fo:04}__LIDAR_TOP.npy'),
                                        pts_occlusion_filter=self.pts_occlusion_filter,
                                        random_angle=random_angle)

        cv2.imwrite("binimg.png", np.array(binimg[0])*255.0)
        cv2.imwrite("sample.png", np.moveaxis(np.array(sample['image'][0]),0,2)*255.0)

        data = Sample(
                view=self.view,
                bev=binimg,
                **sample    
            )
        return data

class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(dataroot, data_aug_conf, grid_conf, parser_name, pitch_adjust=0, limit=None, noadjust=False, multi=False, random_angle=False):
    data = None
    if multi:
        datasets = []
        """ multi format: [[dataroot,pitch,limit],...[]] """
        for i, dataset in enumerate(multi):
            dataroot = dataset[0]
            pitch = dataset[1]
            limit = dataset[2]
            print([dataroot,pitch,limit])
            #if i == 0: continue
            traindata = CarlaBEV2(dataroot, True, data_aug_conf, grid_conf, ret_boxes=False, pts_occlusion_filter=0, pitch_adjust=pitch_adjust,
                    limit=limit,noadjust=noadjust,random_angle=random_angle)
            datasets.append(traindata)

        data = torch.utils.data.ConcatDataset(datasets)
    else:
        data = CarlaBEV2(dataroot, True, data_aug_conf, grid_conf, ret_boxes=False, pts_occlusion_filter=0,pitch_adjust=pitch_adjust,
                limit=limit,noadjust=noadjust,random_angle=random_angle)
    #valdata = CarlaBEV2(dataroot, True, data_aug_conf, grid_conf, ret_boxes=False, pts_occlusion_filter=0, limit=5000)
    return data

    """
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    
    return trainloader, valloader
    """

def compile_test_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    valdata = CarlaBEV2(dataroot, True, data_aug_conf, grid_conf, ret_boxes=False, pts_occlusion_filter=0)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return None, valloader

def compile_viz_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    valdata = CarlaBEV2(dataroot, True, data_aug_conf, grid_conf, ret_boxes=True, pts_occlusion_filter=0)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return None, valloader


if __name__ == "__main__":
    H=225
    W=400
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=6
    max_grad_norm=5.0
    pos_weight=2.13
    logdir='./runs'

    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]

    bsz=4
    nworkers=10
    lr=1e-3
    weight_decay=1e-7
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    #'cams': ['00','01','02','03','04','05'],
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    #'cams': ['CAM_FRONT'],
                    'Ncams': ncams,
                }
