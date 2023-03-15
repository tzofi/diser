from glob import glob
import os
import json
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

from .tools import plot_car, CAMORDER


def parse_calibration(calib, width, height, cam_adjust):
    info = {}
    for k,cal in calib.items():
        trans = [cal['trans'][0], -cal['trans'][1], cal['trans'][2]]

        intrins = np.identity(3)
        intrins[0, 2] = width / 2.0
        intrins[1, 2] = height / 2.0
        intrins[0, 0] = intrins[1, 1] = width / (2.0 * np.tan((cal['fov']+cam_adjust[k]['fov']) * np.pi / 360.0))

        coordmat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        rot = np.matmul(Quaternion(axis=[0, 0, 1], angle=np.radians(-(cal['yaw'] + cam_adjust[k]['yaw']))).rotation_matrix, np.linalg.inv(coordmat))
        quat = Quaternion(matrix=rot)

        info[k] = {'trans': trans, 'intrins': intrins,
                   'rot': quat}
    return info


def parse_nboxes(gt):
    nboxes = []
    for box in gt:
        diffw = box[:3, 1] - box[:3, 2]
        diffl = box[:3, 0] - box[:3, 1]
        diffh = box[:3, 4] - box[:3, 0]

        center = (box[:3, 4] + box[:3, 2]) / 2
        # carla flips y axis
        center[1] = -center[1]

        dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]

        # might need to be transposed?
        rot = np.zeros((3, 3))
        rot[:, 1] = diffw / dims[0]
        rot[:, 0] = diffl / dims[1]
        rot[:, 2] = diffh  / dims[2]

        quat = Quaternion(matrix=rot)
        # again, carla flips y axis
        newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)

        nbox = Box(center, dims, newquat)
        nboxes.append(nbox)

    return nboxes


def visualize(outf='./results', calibf='./nusccalib.json', height=900, width=1600):
    print('reading', calibf)
    with open(calibf, 'r') as reader:
        calib = json.load(reader)
    name2ix = {cam: cami for cami,cam in enumerate(CAMORDER)}
    camixes = [name2ix['CAM_FRONT_LEFT'], name2ix['CAM_FRONT'], name2ix['CAM_FRONT_RIGHT'],
               name2ix['CAM_BACK_LEFT'], name2ix['CAM_BACK'], name2ix['CAM_BACK_RIGHT']]
    
    fac = 5

    folders = sorted(glob(os.path.join(outf, '*')))
    for folderi,folder in enumerate(folders):
        files = glob(os.path.join(folder, '*.jpg'))
        nsteps = max([int(f.split('/')[-1].split('_')[0]) for f in files]) + 1

        gtf = os.path.join(folder, 'info.json')
        print('reading', gtf)
        with open(gtf, 'r') as writer:
            jsgt = json.load(writer)
        calinfo = parse_calibration(calib[jsgt['scene_calib']], width, height, jsgt['cam_adjust'])
        gt = jsgt['boxes']
        
        for step in range(nsteps):
            rat = height / width
            val = 10.25
            fig = plt.figure(figsize=(val + val/3*2*rat - 0.01, val/3*2*rat))
            gs = mpl.gridspec.GridSpec(2, 4, width_ratios=(1, 1, 1, 2*rat))
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

            for vizix,cami in enumerate(camixes):
                imname = os.path.join(folder, f'{step:04}_{cami:02}.jpg')
                showimg = Image.open(imname)
                # if vizix > 2:
                #     showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

                ax = plt.subplot(gs[vizix // 3, vizix % 3])
                plt.imshow(showimg)

                # project 3d points to the camera
                nboxes = parse_nboxes(np.array(gt[step]))
                for box in nboxes:
                    #  Move box to sensor coord system.
                    box.translate(-np.array(calinfo[CAMORDER[cami]]['trans']))
                    box.rotate(Quaternion(calinfo[CAMORDER[cami]]['rot']).inverse)
                    if box_in_image(box, calinfo[CAMORDER[cami]]['intrins'], (width, height), BoxVisibility.ANY):
                        c = np.array([255, 158, 0])/255.0
                        box.render(ax, view=calinfo[CAMORDER[cami]]['intrins'], normalize=True, colors=(c,c,c))
                plt.xlim((0, width))
                plt.ylim((height, 0))
                plt.axis('off')

            # plot bev bboxes
            nboxes = parse_nboxes(np.array(gt[step]))
            ax = plt.subplot(gs[:, 3])
            for box in nboxes:
                pts = box.bottom_corners()[:2].T
                plt.fill(pts[:, 0], pts[:, 1], c='k')
                # plt.fill(box[0, [0, 1, 2, 3]], -box[1, [0, 1, 2, 3]], c='k')
            plot_car((0.0, 0.0, 0.0, 4.68, 1.88), color=(0.0, 0.443, 0.773))
            plt.xlim((-60, 60))
            plt.ylim((-60, 60))
            ax.set_aspect('equal')
            plt.axis('off')

            imname = f'test{folderi:05}_{step:03}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            plt.close(fig)


def copy_map_data(mapnum, outf='/results', totalfnum=4000):
    """Only intended to run on ngc.
    mapnum is the number of maps to use.
    totalfnum is number of episodes in each dataset.
    Maps are chosen like:
    1: Map3
    2: Map3, Map4
    3: Map3, Map4, Map5
    4: Map3, Map4, Map5, Map1
    5: Map3, map4, Map5, Map1, Map2
    """
    ixes = list(range(totalfnum))

    mapordering = [3, 4, 5, 1, 2]
    mapids = []
    for mapi in range(mapnum):
        mapids.extend([
            mapordering[mapi % len(mapordering)]
            for _ in range(int(mapi/mapnum * totalfnum), int((mapi+1)/mapnum * totalfnum))])
    assert(len(mapids) == totalfnum), f'{len(mapids)} {totalfnum}'

    for ix, mapid in zip(ixes, mapids):
        cmd = f'cp -r /mount/carlasim{mapid}/{ix} /results/{ix}_{mapid}'
        print(cmd)
        os.system(cmd)


def find_lyft_run():
    # ~/ngc batch list --begin-time 2020-01-01::00:00:00 --end-time 2020-06-01::00:00:00 --format_type json > findlyft.json
    with open('findlyft.json', 'r') as reader:
        info = json.load(reader)

    for row in info:
        cmd = row['jobDefinition']['command']
        if 'train_seg lyft' in cmd:
            print(row['id'], cmd, '\n')
