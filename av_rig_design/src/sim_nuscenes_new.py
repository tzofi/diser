import json
import os
import sys
import queue
import pygame
import numpy as np
import weakref
from time import sleep, time
from glob import glob

#sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.13-py3.7-linux-x86_64.egg'))
sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.11-py3.7-linux-x86_64.egg'))
sys.path.append(os.environ['CARLAPATH'])
import carla
from pygame.locals import K_0, K_9, K_ESCAPE, K_SPACE, K_d, K_a, K_s, K_w

from .tools import (add_npcs, init_env, ClientSideBoundingBoxes,
                    weather_ps, name2weather, save_data, CAMORDER, bbox_to_2d_lim,
                    determine_filter)
from .annotator import auto_annotate


def render(cameras, display, current_ix, headless, filter_occluded):
    img_data = []
    depth_data_1 = []
    #depth_data_2 = []
    for cam in cameras:
        image = cam['queue'].get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        img_data.append(array.copy())
    if filter_occluded:
        for cam in cameras:
            image = cam['depth_q'].get()
            image.convert(carla.ColorConverter.Depth)
            array = np.array(image.raw_data).reshape((image.height,image.width,4))[:,:,0] * 1000 / 255
            #array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            #array = np.reshape(array, (image.height, image.width, 4))
            #array = array[:, :, :3]
            #array = array[:, :, ::-1]
            depth_data_1.append(array.copy())

    """
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            depth_data_2.append(array.copy())

    if filter_occluded:
        for cam in cameras:
            image = cam['depth_q'].get()
            image.convert(carla.ColorConverter.Depth)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            depth_data_1.append(array.copy())
    """
    if not headless:
        surface = pygame.surfarray.make_surface(img_data[current_ix].swapaxes(0, 1))
        display.blit(surface, (0, 0))
    return img_data, depth_data_1 #, depth_data_2


def camera_blueprint(world, width, height, VIEW_FOV, motion_blur_strength, depth_camera=False):
    """
    Returns camera blueprint.
    """
    bp_name = 'sensor.camera.depth' if depth_camera else 'sensor.camera.rgb'
    camera_bp = world.get_blueprint_library().find(bp_name)
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(VIEW_FOV))

    if motion_blur_strength is not None and not depth_camera:
        print('setting blur', motion_blur_strength)
        camera_bp.set_attribute('motion_blur_intensity', str(motion_blur_strength))
        camera_bp.set_attribute('motion_blur_max_distortion', str(motion_blur_strength))

    return camera_bp


def get_default_cameras(calib, world, width, height, car, motion_blur_strength, cam_adjust, filter_occluded):
    cameras = []
    intrins = []
    for camname in CAMORDER:
        info = calib[camname]
        print(info['fov'])
        camera_transform = carla.Transform(carla.Location(x=info['trans'][0], y=info['trans'][1], z=info['trans'][2]),
                                           carla.Rotation(yaw=info['yaw']+cam_adjust[camname]['yaw'], pitch=0.0, roll=0.0))

        camera = world.spawn_actor(camera_blueprint(world, width, height, info['fov']+cam_adjust[camname]['fov'], motion_blur_strength), camera_transform, attach_to=car)
        if filter_occluded:
            depth_camera = world.spawn_actor(camera_blueprint(world, width, height, info['fov']+cam_adjust[camname]['fov'], motion_blur_strength, depth_camera=True), camera_transform, attach_to=car)

        calibration = np.identity(3)
        calibration[0, 2] = width / 2.0
        calibration[1, 2] = height / 2.0
        calibration[0, 0] = calibration[1, 1] = width / (2.0 * np.tan((info['fov']+cam_adjust[camname]['fov']) * np.pi / 360.0))
        camera.calibration = calibration
        intrins.append(calibration)

        if not filter_occluded:
            cameras.append({'cam': camera, 'queue': queue.Queue()})
        else:
            cameras.append({'cam': camera, 'queue': queue.Queue(), 'depth_cam': depth_camera, 'depth_q': queue.Queue()})
    exit()

    for v in cameras:
        v['cam'].listen(v['queue'].put)
        if filter_occluded:
            v['depth_cam'].listen(v['depth_q'].put)

    intrins = np.stack(intrins, 0)

    return cameras, intrins

def get_cameras(cams, calib, fov, world, width, height, car, motion_blur_strength, filter_occluded):
    cameras = []
    intrins = []
    for i, cam in enumerate(cams):
        cam = [float(c) for c in cam]
        fov = cam[-1]

        """ temporary simplification 
            cam[0] = x
            cam[1] = y
            cam[2] = z
            cam[3] = pitch
            cam[4] = yaw
            cam[5] = fov
        """
        """
        camname = CAMORDER[i]
        info = calib[camname]
        cam[0] = info['trans'][0]
        cam[1] = info['trans'][1]
        cam[2] = info['trans'][2]
        cam[3] = 0.0
        cam[4] = info['yaw']
        """

        camera_transform = carla.Transform(carla.Location(x=cam[0], y=cam[1], z=cam[2]),
                                           carla.Rotation(yaw=cam[4], pitch=cam[3], roll=0.0))

        camera = world.spawn_actor(camera_blueprint(world, width, height, fov, motion_blur_strength), camera_transform, attach_to=car)
        if filter_occluded:
            depth_camera = world.spawn_actor(camera_blueprint(world, width, height, fov, motion_blur_strength, depth_camera=True), camera_transform, attach_to=car)

        calibration = np.identity(3)
        calibration[0, 2] = width / 2.0
        calibration[1, 2] = height / 2.0
        calibration[0, 0] = calibration[1, 1] = width / (2.0 * np.tan((fov) * np.pi / 360.0))
        camera.calibration = calibration
        intrins.append(calibration)

        if not filter_occluded:
            cameras.append({'cam': camera, 'queue': queue.Queue()})
        else:
            cameras.append({'cam': camera, 'queue': queue.Queue(), 'depth_cam': depth_camera, 'depth_q': queue.Queue()})

    for v in cameras:
        v['cam'].listen(v['queue'].put)
        if filter_occluded:
            v['depth_cam'].listen(v['depth_q'].put)

    intrins = np.stack(intrins, 0)

    return cameras, intrins


def scrape_single(world, car_bp, ego_start, clock, display, nnpcs,
                  pos_agents, pos_inits, weather_name, pref, width, height,
                  calib, current_ix, start_ix, headless, car_color_range,
                  og_color, nnpc_position_std, nnpc_yaw_std, motion_blur_strength,
                  cam_adjust, filter_occluded, skipframes, cameras, fov, tm_port, num_npcs, npcs, car):
    world.set_weather(weather_name)
    car.set_autopilot(enabled=True, tm_port=tm_port)

    # SETUP CAMERAS
    intrins = []
    if cameras is not None:
        cameras, intrins = get_cameras(cameras, calib, fov, world, width, height, car, motion_blur_strength, filter_occluded)
    else:
        cameras, intrins = get_default_cameras(calib, world, width, height, car, motion_blur_strength, cam_adjust, filter_occluded)
    

    def destroy():
        for cam in cameras:
            cam['cam'].destroy()
            if 'depth_cam' in list(cam.keys()):
                cam['depth_cam'].destroy()
    
    try:
        images = []
        boxes_1 = []
        boxes_2 = []
        for prestep in range(pref):
            # event handler
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP and event.key >= K_0 and event.key <= K_9:
                        current_ix = (event.key - K_0) % len(cameras)
                        print(list(calib.keys())[current_ix])
            world.tick()

            if prestep % skipframes != 0: continue

            img_data, depth_data_1 = render(cameras, display, current_ix, headless, filter_occluded)
            for i, img in enumerate(img_data):
                img_data[i] = np.rollaxis(img, 2, 0).astype(np.float32) / 255.0
            img_data = np.array(img_data)

            # bounding boxes
            bboxes = ClientSideBoundingBoxes.get_global_bbox(npcs)

            # filter occluded boxes
            if filter_occluded:
                #dont_filter = determine_filter(bboxes, cameras, depth_data_1)
                #bboxes_1 = [bboxes[i] for i in range(len(bboxes)) if dont_filter[i]]
                bboxes_1 = bboxes

                dont_filters = []
                for cam_idx in range(len(cameras)):
                    filtered_out, removed_out, dont_filter = auto_annotate(npcs, cameras[cam_idx]['cam'], depth_data_1[cam_idx])
                    dont_filters.append(dont_filter)
                
                master_dont_filter = {}
                for idx in range(len(dont_filters[0])):
                    df = 0
                    for cam_idx in range(len(cameras)):
                        df += int(dont_filters[cam_idx][idx]) 
                        if df > 0: break
                    master_dont_filter[idx] = bool(df)
                bboxes_2 = [bboxes[i] for i in range(len(bboxes)) if master_dont_filter[i]]



            #bounding_boxes_1 = ClientSideBoundingBoxes.get_camera_boxes(bboxes_1, cameras[current_ix]['cam'])
            #bounding_boxes_2 = ClientSideBoundingBoxes.get_camera_boxes(bboxes_2, cameras[current_ix]['cam'])

            if prestep >= start_ix and prestep % skipframes == 0:
                car_bboxes_1 = np.array([
                    np.dot(np.linalg.inv(
                        ClientSideBoundingBoxes.get_matrix(car.get_transform())
                        ), bbox)
                    for bbox in bboxes_1
                ]).tolist()
                car_bboxes_2 = np.array([
                    np.dot(np.linalg.inv(
                        ClientSideBoundingBoxes.get_matrix(car.get_transform())
                        ), bbox)
                    for bbox in bboxes_2
                ]).tolist()
                images.append(img_data)
                boxes_1.append(car_bboxes_1)
                boxes_2.append(car_bboxes_2)
                #data.append({
                #    'imgs': img_data,
                #    'car_bboxes': car_bboxes,
                #})
    finally:
        destroy()
    
    images = np.stack(images, 0)
    return current_ix, images, intrins, boxes_1, boxes_2

class CarlaClient:
    def __init__(self, host='127.0.0.1', port=2000, tm_port=8000, width=400, height=224, timeout=30.0, 
            ntrials=1, pref=100, calibf='./nusccalib.json', outf=None, start_ix=0, 
            ncarcalib='./nuscncars.json', rnd_seed=0, headless=True, skipframes=10, bs=1, num_npcs=15,
           map_name='Town03',  # set the map (01-05)

           fix_nnpcs=None,  # set to an integer to fix the number of npcs per scene (0 or greater int)
           uniform_nnpcs=False,  # set true to uniformly sample from the number of npcs

           p_assets=1.0,  # fraction of assets to use (float between 0 and 1)

           car_color_range=0.5,  # each color will be Unif(0.5-car_color_range, 0.5+car_color_range) (float between 0 and 0.5)
           og_color=False,  # set true to leave the color to the default

           weather_max=None,  # weather parameters will be Unif(0, weather_max*100) if set (float between 0 and 1)

           nnpc_position_std=0.0,  # (meters) npc initial position uniform noise half-width (float non-negative)

           nnpc_yaw_std=0.0,  # (degrees) npc initial heading uniform noise half-width (float non-negative)

           motion_blur_strength=None,  # determines amount of motion blur (float between 0 and 1)

           cam_yaw_adjust=0.0,  # (degrees) uniform half-width
           cam_fov_adjust=0.0,  # (degrees) uniform half-width

           filter_occluded=True,

           fov=64.5614792846793,
           ):

        if bs == 1:
            skipframes=1
            pref=1
        else:
            pref = skipframes * bs

        self.num_npcs = num_npcs
        self.fix_nnpcs= fix_nnpcs
        self.uniform_nnpcs = uniform_nnpcs
        self.weather_max = weather_max
        self.cam_yaw_adjust = cam_yaw_adjust
        self.cam_fov_adjust = cam_fov_adjust
        self.filter_occluded = filter_occluded
        self.pref = pref
        self.width = width
        self.height = height
        self.start_ix = start_ix
        self.car_color_range = car_color_range
        self.og_color = og_color
        self.nnpc_position_std = nnpc_position_std
        self.nnpc_yaw_std = nnpc_yaw_std
        self.motion_blur_strength = motion_blur_strength
        self.skipframes = skipframes
        self.fov = fov
        self.tm_port = tm_port

        #np.random.seed(rnd_seed)
        self.pos_inits, self.pos_agents, self.world, self.calib, self.car_bp, self.pos_weathers, self.scene_names, self.ncarinfo, self.client = \
                init_env(host, port, tm_port, width, height, timeout, calibf, ncarcalib, rnd_seed, map_name, p_assets)
        self.display, self.clock = None, None

        weather_prob = [weather_ps[i] for i in range(len(self.pos_weathers))]
        self.weather_prob = np.array(weather_prob) / sum(weather_prob)
        self.current_ix = 0  # current viewable camera
        self.trial_ixes = list(range(ntrials))

        self.init_ix = np.random.randint(len(self.pos_inits))
        if self.fix_nnpcs is not None:
            self.nnpc = self.fix_nnpcs
        else:
            if self.uniform_nnpcs:
                #print('nnpcs uniform')
                self.nnpc = np.random.choice(self.ncarinfo[:, 0])
            else:
                #print('nnpcs nuscenes distribution')
                self.nnpc = np.random.choice(self.ncarinfo[:, 0], p=self.ncarinfo[:, 1] / self.ncarinfo[:, 1].sum())

        if self.weather_max is not None:
            #print('weather random', self.weather_max)
            self.chosen_weather = carla.WeatherParameters(cloudiness=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     precipitation=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     precipitation_deposits=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     wind_intensity=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     wetness=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     fog_density=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     sun_altitude_angle=np.random.uniform(-90.0, 0.0) if np.random.uniform(0.0, 1.0) < 0.1 else np.random.uniform(0.0, 90.0*(1.0 - self.weather_max))
                                                     )
        else:
            #print('weather categorical')
            weather_ix = int(np.random.choice(list(range(len(self.pos_weathers))), p=self.weather_prob))
            self.chosen_weather = name2weather[self.pos_weathers[weather_ix]]
        self.calib_ix = np.random.randint(len(self.calib))
        self.cam_adjust = {cam: {'yaw': np.random.uniform(-self.cam_yaw_adjust, self.cam_yaw_adjust),
                            'fov': np.random.uniform(-self.cam_fov_adjust, self.cam_fov_adjust)} for cam in CAMORDER}

        self.cnt = 0
        self.update_scene()

    def update_scene(self):
        if self.cnt > 0:
            self.car.destroy()
            for npc in self.npcs:
                npc.destroy()
        self.init_ix = np.random.randint(len(self.pos_inits))
        if self.fix_nnpcs is not None:
            self.nnpc = self.fix_nnpcs
        else:
            if self.uniform_nnpcs:
                #print('nnpcs uniform')
                self.nnpc = np.random.choice(self.ncarinfo[:, 0])
            else:
                #print('nnpcs nuscenes distribution')
                self.nnpc = np.random.choice(self.ncarinfo[:, 0], p=self.ncarinfo[:, 1] / self.ncarinfo[:, 1].sum())

        if self.weather_max is not None:
            #print('weather random', self.weather_max)
            self.chosen_weather = carla.WeatherParameters(cloudiness=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     precipitation=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     precipitation_deposits=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     wind_intensity=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     wetness=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     fog_density=np.random.uniform(0.0, 100.0*self.weather_max),
                                                     sun_altitude_angle=np.random.uniform(-90.0, 0.0) if np.random.uniform(0.0, 1.0) < 0.1 else np.random.uniform(0.0, 90.0*(1.0 - self.weather_max))
                                                     )
        else:
            #print('weather categorical')
            weather_ix = int(np.random.choice(list(range(len(self.pos_weathers))), p=self.weather_prob))
            self.chosen_weather = name2weather[self.pos_weathers[weather_ix]]
        self.calib_ix = np.random.randint(len(self.calib))
        self.cam_adjust = {cam: {'yaw': np.random.uniform(-self.cam_yaw_adjust, self.cam_yaw_adjust),
                            'fov': np.random.uniform(-self.cam_fov_adjust, self.cam_fov_adjust)} for cam in CAMORDER}

        self.car_bp.set_attribute('color', f'{np.random.randint(256)},{np.random.randint(256)},{np.random.randint(256)}')
        map = self.world.get_map()
        success = False
        self.car = None
        location = None
        while not success:
            try:
                location = np.random.choice(self.pos_inits)
                wp = map.get_waypoint(location.location)
                location = wp.transform
                location.location.z = location.location.z + 0.05
                self.car = self.world.spawn_actor(self.car_bp, location)
                success = True
            except Exception as e:
                print(e)
                pass
        self.npcs = add_npcs(self.world, self.num_npcs, self.pos_agents, location, self.pos_inits, self.car_color_range, self.og_color,
                        self.nnpc_position_std, self.nnpc_yaw_std, self.tm_port)
        self.cnt += 1

    def rollout(self, cameras=None):
        sessions = []
        success = False
        i = 0
        while not success:
            i += 1
            try:
                sessions = []
                for trial in self.trial_ixes:
                    t0 = time()
                    #init_ix = np.random.randint(len(self.pos_inits))

                    self.current_ix, images, intrins, bboxes_1, bboxes_2 = scrape_single(self.world, self.car_bp, self.pos_inits[self.init_ix], self.clock, self.display, self.nnpc,
                                                            self.pos_agents, self.pos_inits, self.chosen_weather, self.pref, self.width, self.height,
                                                            self.calib[self.calib_ix], self.current_ix, self.start_ix, True, self.car_color_range,
                                                            self.og_color, self.nnpc_position_std, self.nnpc_yaw_std, self.motion_blur_strength,
                                                            self.cam_adjust, self.filter_occluded, self.skipframes, cameras, self.fov, self.tm_port, self.num_npcs, self.npcs, self.car)
                    sessions.append([images, intrins, bboxes_1, bboxes_2])
                    t1 = time()
                    print('finished episode number', trial, 'in time', t1 - t0)
                success = True
            except Exception as e:
                self.update_scene()
                if i > 1000:
                    print("Stuck!")
                    print(e)
                    exit()
                pass
        return sessions


if __name__ == "__main__":
    c = CarlaClient()
    c.rollout()
    sessions = c.rollout(cameras=[[1.0,1.0,0.0,0.0],
                                  [1.0,1.0,0.0,0.0]])
    print(len(sessions))
    print(len(sessions[0][0]))
    print(sessions[0][0][0].shape)

    sessions = c.rollout(cameras=[[1.0,1.0,0.0,0.0],
                                  [1.0,1.0,0.0,0.0]])
    print(len(sessions))
    print(len(sessions[0][0]))
    print(sessions[0][0][0].shape)

def scrape(host='127.0.0.1', port=2000, width=400, height=225, timeout=30.0, ntrials=250, pref=150,
           calibf='./nusccalib.json', outf=None, start_ix=50, ncarcalib='./nuscncars.json',
           rnd_seed=42, headless=False, skipframes=10,

           map_name='Town03',  # set the map (01-05)

           fix_nnpcs=2,  # set to an integer to fix the number of npcs per scene (0 or greater int)
           uniform_nnpcs=False,  # set true to uniformly sample from the number of npcs

           p_assets=1.0,  # fraction of assets to use (float between 0 and 1)

           car_color_range=0.5,  # each color will be Unif(0.5-car_color_range, 0.5+car_color_range) (float between 0 and 0.5)
           og_color=False,  # set true to leave the color to the default

           weather_max=None,  # weather parameters will be Unif(0, weather_max*100) if set (float between 0 and 1)

           nnpc_position_std=0.0,  # (meters) npc initial position uniform noise half-width (float non-negative)

           nnpc_yaw_std=0.0,  # (degrees) npc initial heading uniform noise half-width (float non-negative)

           motion_blur_strength=None,  # determines amount of motion blur (float between 0 and 1)

           cam_yaw_adjust=0.0,  # (degrees) uniform half-width
           cam_fov_adjust=0.0,  # (degrees) uniform half-width

           filter_occluded=False,
           ):
    np.random.seed(rnd_seed)
    pos_inits, pos_agents, world, calib, car_bp, pos_weathers, scene_names, ncarinfo, client = init_env(host, port, width, height, timeout, calibf,
                                                                                                        ncarcalib, rnd_seed, map_name, p_assets)

    if headless:
        display, clock = None, None
    else:
        pygame.init()
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()

    weather_prob = [weather_ps[i] for i in range(len(pos_weathers))]
    weather_prob = np.array(weather_prob) / sum(weather_prob)
    current_ix = 0  # current viewable camera

    trial_ixes = list(range(ntrials))
    # check folders that have already been made
    if outf is not None:
        trial_fs = glob(os.path.join(outf, '*'))
        trial_ixes = [ix for ix in trial_ixes if os.path.join(outf, str(ix)) not in trial_fs]
        print('SOME FILES MIGHT ALREADY EXIST!!!!!!!!!! ntrials:', ntrials, 'num left:', len(trial_ixes))

    for trial in trial_ixes:
        t0 = time()
        init_ix = np.random.randint(len(pos_inits))

        # nnpcs
        if fix_nnpcs is not None:
            nnpc = fix_nnpcs
        else:
            if uniform_nnpcs:
                print('nnpcs uniform')
                nnpc = np.random.choice(ncarinfo[:, 0])
            else:
                print('nnpcs nuscenes distribution')
                nnpc = np.random.choice(ncarinfo[:, 0], p=ncarinfo[:, 1] / ncarinfo[:, 1].sum())

        if weather_max is not None:
            print('weather random', weather_max)
            chosen_weather = carla.WeatherParameters(cloudiness=np.random.uniform(0.0, 100.0*weather_max),
                                                     precipitation=np.random.uniform(0.0, 100.0*weather_max),
                                                     precipitation_deposits=np.random.uniform(0.0, 100.0*weather_max),
                                                     wind_intensity=np.random.uniform(0.0, 100.0*weather_max),
                                                     wetness=np.random.uniform(0.0, 100.0*weather_max),
                                                     fog_density=np.random.uniform(0.0, 100.0*weather_max),
                                                     sun_altitude_angle=np.random.uniform(-90.0, 0.0) if np.random.uniform(0.0, 1.0) < 0.1 else np.random.uniform(0.0, 90.0*(1.0 - weather_max))
                                                     )
        else:
            print('weather categorical')
            weather_ix = int(np.random.choice(list(range(len(pos_weathers))), p=weather_prob))
            chosen_weather = name2weather[pos_weathers[weather_ix]]

        calib_ix = np.random.randint(len(calib))
        cam_adjust = {cam: {'yaw': np.random.uniform(-cam_yaw_adjust, cam_yaw_adjust),
                            'fov': np.random.uniform(-cam_fov_adjust, cam_fov_adjust)} for cam in CAMORDER}
        print('cam_adjust:', cam_adjust)

        current_ix, data = scrape_single(world, car_bp, pos_inits[init_ix], clock, display, nnpc,
                                        pos_agents, pos_inits, chosen_weather, pref, width, height,
                                        calib[calib_ix], current_ix, start_ix, headless, car_color_range,
                                        og_color, nnpc_position_std, nnpc_yaw_std, motion_blur_strength,
                                        cam_adjust, filter_occluded)

        if outf is not None:
            save_data(data, outf, trial, scene_names[calib_ix], skipframes, cam_adjust)
        t1 = time()
        print('finished episode number', trial, 'in time', t1 - t0)
    if not headless:
        pygame.quit()
