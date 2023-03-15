import glob
from PIL import Image

import torch
import numpy as np
import random
# from stable_baselines.common import set_global_seeds
import get_premade_scenes
import matplotlib.pyplot as plt

import gym
from gym import spaces
from copy import deepcopy
import warnings
import utils 

def create_P(camera):
    P = torch.concat([camera.camera.cpu(), camera.eye_pos.cpu().unsqueeze(1)], 1)
    P = torch.concat([P, torch.tensor([0,0,0,1]).unsqueeze(0)]).float()
    return P 

def project_coord_into_image(P, coord):
    return torch.inverse(P) @ coord.T

def get_reward_for_projected_verts(verts, camera, res, image=None, normalize_reward=True):
    P = create_P(camera)
    verts = torch.concat([verts, torch.ones(verts.shape[0]).unsqueeze(1)], 1).float()
    cam_coords = []
    rewards = []
    total_reward = 0.
    for i in range(len(verts)):
        cam_coord = project_coord_into_image(P, verts[i])
        x = (cam_coord[1]/cam_coord[2]).int()
        y = (cam_coord[0]/cam_coord[2]).int()
        # NOTE: case when half of the vertices are hidden and the other half visible, reward = 0 
        if x >= 0 and x < res and y >= 0 and y < res:
            rewards.append(1.0)
            total_reward += 1.0
            if image is not None:
                image[x, y,:] = 0
        else:
            rewards.append(-1.0)
            total_reward += -1.0
            
        cam_coords.append(cam_coord)
    if normalize_reward:
        total_reward = total_reward/len(verts)
    return cam_coords, rewards, total_reward, image

def get_bounding_boxes(verts, camera, res, image=None, draw_bbox=None):
    P = create_P(camera)
    verts = torch.concat([verts, torch.ones(verts.shape[0]).unsqueeze(1)], 1).float()
    cam_coords = []
    for i in range(len(verts)):
        cam_coord = project_coord_into_image(P, verts[i])
        x = torch.clip((cam_coord[1]/cam_coord[2]).int(), min=0., max=res)
        y = torch.clip((cam_coord[0]/cam_coord[2]).int(), min=0., max=res)
        cam_coords.append([x,y])

    cam_coords = torch.tensor(cam_coords)    
    minx, _ = torch.min(cam_coords[:,0], axis=0)
    maxx, _ = torch.max(cam_coords[:,0], axis=0)
    miny, _ = torch.min(cam_coords[:,1], axis=0)
    maxy, _ = torch.max(cam_coords[:,1], axis=0)
    top_left = (miny.item(), minx.item())
    top_right = (maxy.item(), maxx.item())
    if draw_bbox: 
        if image is not None: 
            image = to_uint8(image.detach().cpu()).numpy()
            image = cv2.rectangle(image, top_left, top_right, (255,0,0), 1)
    
    return cam_coords, top_left, top_right, image

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    # (x1,y1,x2,y2)
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def rotate_around_z(pos, theta):
    x,y,z = pos
    # theta = theta - 90.
    theta = -theta
    theta = theta * np.pi/180.
    x_p = x*np.cos(theta) - z * np.sin(theta)
    z_p = x*np.sin(theta) + z * np.cos(theta)
    return np.array([x_p,y,z_p])

# def rotate_around_z(pos, theta):
#     x,y,z = pos
#     # theta = theta - 90.
#     theta = -theta
#     theta = theta * np.pi/180.
#     rot = torch.tensor([
#         [torch.cos(theta), 0 , torch.sin(theta)],
#         [0, 1, 0],
#         [torch.sin(theta), 0 , torch.cos(theta)],
#     ])
#     return pos_p * rot
#     x_p = x * torch.cos(theta) - z * torch.sin(theta)
#     z_p = x * torch.sin(theta) + z * torch.cos(theta)
#     return torch.tensor([x_p,y,z_p])

def get_lookout_from_angles(pos, theta, phi):
    theta = theta * np.pi/180.
    phi = phi * np.pi/180.
    x = pos[0] + np.cos(theta) 
    z = pos[2] + np.sin(theta)
    y = pos[1] + np.cos(phi)
    # z = ((pos[2] + np.sin(phi)) + z )/2. 
    # print(x,y,z)
    return np.array([x,y,z])

def check_bounds_vector(state, bounds):
    min_, max_ = bounds
    for i in range(len(state)):
        if state[i] > max_: 
            state[i] = max_
        if state[i] < min_ :
            state[i] = min_
    return state

def check_bounds_primitive_type(state, bounds):
    min_, max_ = bounds
    if state > max_: 
        state = max_
    if state < min_ :
        state = min_
    return state

def make_gif(imgs, fp_out):
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
            save_all=True, loop=0)

def perform_inference(model, env, plot_figure=True, gif_path=None):
    observations = []
    rewards = []
    dones = []
    obs = env.reset()
    for i in range(env.rollout_length):
        action, _states = model.predict(obs)
        if plot_figure:
            plt.figure(); plt.imshow(obs)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)

    if gif_path is not None: 
        make_gif(observations, gif_path)
    return observations, rewards, dones

def get_new_imgs(images, new_max=None, rescale=False):
    new_imgs = []
    for i in images.keys():
        if not isinstance(i, int):
            continue
        img = np.array(images[i])
        k = deepcopy(img)
        k = k.reshape(-1)
        k.sort()
        print("image maximum", k[-10:-1])
        if new_max is None: 
    #         print(k[-1], k[-2])
            new_max = np.minimum(k[-2]+10, k[-1])
    #         print(new_max)
        print("threshold at", new_max)

        if rescale: 
            e_min = 0
            e_max = new_max
            new_img = (2*(np.minimum(img, e_max) - e_min) / (e_max - e_min) - 1) # scale it to [-1,1]
            new_img += 1 
        else:
            new_img = np.minimum(new_max, img)
        new_imgs.append(new_img)
    
    return new_imgs

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=15):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'plasma')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()
