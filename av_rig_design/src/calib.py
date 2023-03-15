from nuscenes.nuscenes import NuScenes
import json
import os
import sys
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from collections import Counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.5-py3.5-linux-x86_64.egg'))
sys.path.append(os.environ['CARLAPATH'])
import carla


def get_scenes(version, is_train):
    # filter by scene split
    split = {
        'v1.0-trainval': {True: 'train', False: 'val'},
        'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
    }[version][is_train]

    scenes = create_splits_scenes()[split]

    return scenes


def scrape_calib(version, dataroot='/media/jphilion/52a4e0ca-6b1d-45f6-bb4d-516d3d7b316d/data/data/nuscenes/',
                 outname='nusccalib.json', width=1600, height=900):
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=os.path.join(dataroot, version), verbose=True)

    scenes = get_scenes(nusc.version, is_train=True)

    samples = [samp for samp in nusc.sample if
               nusc.get('scene', samp['scene_token'])['name'] in scenes]

    scene2rig = {}
    for sample in samples:
        scenename = nusc.get('scene', sample['scene_token'])['name']
        if scenename in scene2rig:
            continue
        rig = {}
        for k,v in sample['data'].items():
            if 'CAM' in k:
                token = nusc.get('sample_data', v)['calibrated_sensor_token']
                calib = nusc.get('calibrated_sensor', token)

                # fov
                assert(calib['camera_intrinsic'][0][0] == calib['camera_intrinsic'][1][1]), cam['camera_intrinsic']
                fov = np.degrees(2.0*np.arctan(width/(2.0*calib['camera_intrinsic'][0][0])))

                # yaw
                quat = Quaternion(calib['rotation'])
                mat = quat.rotation_matrix
                newmat = np.matmul(mat, np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
                newquat = Quaternion(matrix=newmat)
                yaw,pitch,roll = newquat.yaw_pitch_roll
                carla_yaw = -np.degrees(yaw)
                assert(np.abs(np.degrees(pitch)) < 1.0 and np.abs(np.degrees(roll)) < 1.0), f'{np.degrees(pitch)} {np.degrees(roll)}'

                # translation
                carla_trans = [calib['translation'][0], -calib['translation'][1], calib['translation'][2]]

                rig[k] = {'fov': fov, 'yaw': carla_yaw, 'trans': carla_trans}
        scene2rig[scenename] = rig

    print('saving', outname)
    with open(outname, 'w') as writer:
        json.dump(scene2rig, writer)


def scrape_ncars(version, dataroot='/media/jphilion/52a4e0ca-6b1d-45f6-bb4d-516d3d7b316d/data/data/nuscenes/',
                 outname='nuscncars.json'):
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=os.path.join(dataroot, version), verbose=True)

    scenes = get_scenes(nusc.version, is_train=True)

    samples = [samp for samp in nusc.sample if
               nusc.get('scene', samp['scene_token'])['name'] in scenes]

    ncars = []
    for sample in samples:
        counter = 0
        # egopose = nusc.get('ego_pose',
        #                    nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        # trans = -np.array(egopose['translation'])
        # rot = Quaternion(egopose['rotation']).inverse
        for tok in sample['anns']:
            inst = nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            # box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            # box.translate(trans)
            # box.rotate(rot)
            counter += 1
        ncars.append(counter)
    
    c = Counter(ncars)
    print(c)
    print('saving', outname)
    with open(outname, 'w') as writer:
        json.dump(c, writer)


def plot_ncardist(outname='nuscncars.json', calibf='./nusccalib.json', imname='ncarhist.jpg'):
    print('reading', outname)
    with open(outname, 'r') as reader:
        data = json.load(reader)
        data = [(int(k),v) for k,v in data.items()]
    data = np.array(sorted(data))

    with open(calibf, 'r') as reader:
        camcal = json.load(reader)
        fovs = []
        yaws = []
        for scene,info in camcal.items():
            for name,cam in info.items():
                fovs.append(cam['fov'])
                yaws.append(cam['yaw'])

    fig = plt.figure(figsize=(15, 5))
    gs = mpl.gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0, 0])
    plt.title('Number of Cars Per Scene')
    plt.bar(data[:, 0], data[:, 1])
    plt.xlabel('Number of Vehicles in a Scene')
    plt.ylabel('Counts')

    ax = plt.subplot(gs[0, 1])
    plt.title('Camera Field of View (intrinsics)')
    plt.hist(fovs, bins=200)
    plt.xlabel('Field of View (degrees)')
    plt.ylabel('Counts')

    ax = plt.subplot(gs[0, 2])
    plt.title('Camera Yaw (extrinsics)')
    plt.hist(yaws, bins=200)
    plt.xlabel('Camera Yaw (degrees)')
    plt.ylabel('Counts')

    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)



def check_parsing(outname='./nusccalib.json'):
    with open(outname, 'r') as reader:
        calib = json.load(reader)
    
    for cam in calib:
        # yaw,pitch,roll = Quaternion(cam['rotation']).yaw_pitch_roll
        # trans = carla.Transform(carla.Location(x=cam['translation'][0], y=cam['translation'][1], z=cam['translation'][2]),
        #                                        carla.Rotation(yaw=-np.degrees(yaw), pitch=np.degrees(pitch), roll=np.degrees(roll)))

        # # print(cam['translation'])
        # print( trans.transform(carla.Location(1.0, 0.0, 0.0)) - trans.transform(carla.Location(0.0, 0.0, 0.0)) )
        # print( trans.transform(carla.Location(0.0, 1.0, 0.0)) - trans.transform(carla.Location(0.0, 0.0, 0.0)) )
        # print( trans.transform(carla.Location(0.0, 0.0, 1.0)) - trans.transform(carla.Location(0.0, 0.0, 0.0)) )
        # print( Quaternion(cam['rotation']).rotation_matrix )
        # print(cam['name'], cam['translation'])
        # print()

        quat = Quaternion(cam['rotation'])
        mat = quat.rotation_matrix
        newmat = np.matmul(mat, np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
        newquat = Quaternion(matrix=newmat)
        print(cam['name'], np.degrees(newquat.yaw_pitch_roll).tolist())
