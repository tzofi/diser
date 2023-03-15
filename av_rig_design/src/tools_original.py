import os
import sys
import numpy as np
import pygame
import json
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.5-py3.5-linux-x86_64.egg'))
sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.8-py3.5-linux-x86_64.egg'))
sys.path.append(os.environ['CARLAPATH'])
import carla


name2weather = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.CloudyNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetCloudyNoon,
    4: carla.WeatherParameters.MidRainyNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.SoftRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.CloudySunset,
    9: carla.WeatherParameters.WetSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    11: carla.WeatherParameters.MidRainSunset,
    12: carla.WeatherParameters.HardRainSunset,
    13: carla.WeatherParameters.SoftRainSunset,
    14: carla.WeatherParameters(
                                cloudiness=80.0,
                                precipitation=30.0,
                                sun_altitude_angle=-90.0),
}


weather_ps = {
    0: 0.25,
    1: 1./24,
    2: 1./24,
    3: 1./24,
    4: 1./24,
    5: 1./24,
    6: 1./24,
    7: 0.25,
    8: 1./24,
    9: 1./24,
    10: 1./24,
    11: 1./24,
    12: 1./24,
    13: 1./24,
    14: 0.25,
}


def tr_dist(t0, t1):
    return np.sqrt((t0.location.x - t1.location.x)**2 + (t0.location.y - t1.location.y)**2 + (t0.location.z - t1.location.z)**2)


def draw_image(image, surface):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    if not surface is None:
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        surface.blit(image_surface, (0, 0))
    return array


def add_npcs(world, nnpcs, pos_agents, ego_start, pos_inits, car_color_range, og_color,
             nnpc_position_std, nnpc_yaw_std, tm_port):
    #print(f'start adding {nnpcs} agents')
    dists = [tr_dist(ego_start, init) for init in pos_inits]
    ixes = sorted(range(len(pos_inits)), key=lambda ix: dists[ix] if dists[ix] > 5 else 10000)

    colorrange = (0.5 - car_color_range, 0.5 + car_color_range)
    #print('npc color range:', colorrange)
    #print('loc std:', nnpc_position_std)
    #print('yaw std:', nnpc_yaw_std)

    agents = []
    #print('adding agents')
    for ix in ixes:
        if len(agents) == nnpcs:
            break
        if np.random.uniform() > 0.05:
            continue
        try:
            agentbp = np.random.choice(pos_agents)
            if agentbp.has_attribute('color') and not og_color:
                rgb = np.random.uniform(colorrange[0], colorrange[1], (3,)) * 255.0
                rgb = [int(c) for c in rgb]
                agentbp.set_attribute('color', f'{rgb[0]},{rgb[1]},{rgb[2]}')
            location = pos_inits[ix]
            loc = carla.Location(location.location.x,
                                location.location.y,
                                location.location.z)
            map = world.get_map()
            location1 = map.get_waypoint(loc)
            location1 = location1.transform
            newlocation = carla.Transform(carla.Location(location1.location.x + np.random.uniform(-nnpc_position_std, nnpc_position_std),
                                                         location1.location.y + np.random.uniform(-nnpc_position_std, nnpc_position_std),
                                                         location1.location.z + 0.05),
                                          carla.Rotation(pitch=location.rotation.pitch,
                                                         yaw=location.rotation.yaw+np.random.uniform(-nnpc_yaw_std, nnpc_yaw_std),
                                                         roll=location.rotation.roll))
            agent = world.spawn_actor(agentbp, newlocation)
            agent.set_autopilot(True, tm_port=tm_port)
            agents.append(agent)
        except Exception as e:
            print(e)
            pass
            #print('Skipping because:', e)
    #print('added agents')
    return agents


def init_env(host, port, tm_port, width, height, timeout, calibf, ncarcalib, rnd_seed,
             map_name, p_assets):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    print('loading map', map_name)
    client.load_world(map_name)
    world = client.get_world()

    settings = world.get_settings()
    tm = client.get_trafficmanager(tm_port)
    tm.global_percentage_speed_difference(0.0)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    tm.set_synchronous_mode(True)
    world.apply_settings(settings)
    world.tick()

    ma = world.get_map()
    bprint_library = world.get_blueprint_library()

    pos_inits = [tr.transform for tr in ma.generate_waypoints(1.0)]
    # avoid collision with ground
    for tr in pos_inits:
        tr.location.z += 2.0
    
    pos_agents = bprint_library.filter('vehicle.*')

    # 1 renders most similarly to renault zoe (nuscenes car)
    # https://en.wikipedia.org/wiki/Renault_Zoe
    car_bp = pos_agents[1]

    # restrict what kinds of agents are used
    split_ix = int(len(pos_agents) * p_assets)
    #print('num agents before:', len(pos_agents), split_ix)
    pos_agents = [pos_agents[i] for i in range(split_ix)]
    #print('num agents after:', len(pos_agents))

    pos_weathers = list(name2weather.keys())

    #print('reading', calibf)
    with open(calibf, 'r') as reader:
        calib = json.load(reader)
    scene_names = list(calib.keys())
    calib = list(calib.values())

    with open(ncarcalib, 'r') as reader:
        ncarinfo = json.load(reader)
        ncarinfo = np.array([(int(k),v) for k,v in ncarinfo.items()])

    return pos_inits, pos_agents, world, calib, car_bp, pos_weathers, scene_names, ncarinfo, client


def save_data(data, outf, trial, scene_name, skipframes, cam_adjust):
    print('saving', trial)
    newf = os.path.join(outf, str(trial))
    os.mkdir(newf)
    jsf = os.path.join(newf, 'info.json')
    with open(jsf, 'w') as writer:
        info = {'boxes': [row['car_bboxes'] for row in data[::skipframes]], 'scene_calib': scene_name,
                'cam_adjust': cam_adjust}
        json.dump(info, writer)
    for rowi,row in enumerate(data[::skipframes]):
        for imgi,img in enumerate(row['imgs']):
            imname = os.path.join(newf, f'{rowi:04}_{imgi:02}.jpg')
            img = Image.fromarray(img)
            img.save(imname)


def bbox_to_2d_lim(bbox, H, W):
    xs = [int(bbox[i, 0]) for i in range(8)]
    ys = [int(bbox[i, 1]) for i in range(8)]
    if max(xs) < 0 or max(ys) < 0 or min(xs) >= W or min(ys) >= H:
        return None, None
    xlim = (max(min(xs), 0), min(max(xs), W))
    ylim = (max(min(ys), 0), min(max(ys), H))
    if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        return None, None
    return xlim, ylim


def determine_filter(bboxes, cameras, depth_data):
    dont_filter = [True for _ in range(len(bboxes))]
    for depi, dep in enumerate(depth_data):
        in_cam_boxes, kept_ixes = ClientSideBoundingBoxes.get_camera_boxes(bboxes, cameras[depi]['cam'], True)
        depth_meters = (dep[:, :, 2]*1.0 + dep[:, :, 1] * 256.0 + dep[:, :, 0] * 256.0 * 256.0) / (256 * 256 * 256 - 1) * 1000.0
        for boxi,box in enumerate(in_cam_boxes):
            xlim, ylim = bbox_to_2d_lim(box, depth_meters.shape[0], depth_meters.shape[1])
            if xlim is not None:
                max_depth = np.max(depth_meters[ylim[0]:ylim[1], xlim[0]:xlim[1]])
                box_min = min([box[i, 2] for i in range(8)])
                if max_depth < box_min:
                    dont_filter[kept_ixes[boxi]] = False
    return dont_filter


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/client_bounding_boxes.py
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, VIEW_WIDTH, VIEW_HEIGHT, BB_COLOR):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bboxi,bbox in enumerate(bounding_boxes):
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])

            # for debugging the occlusion filtering
            # xlim, ylim = bbox_to_2d_lim(bbox, 900, 1600)
            # if xlim is not None:
            #     # draw box
            #     pygame.draw.line(bb_surface, BB_COLOR, (xlim[0], ylim[0]), (xlim[0], ylim[1]))
            #     pygame.draw.line(bb_surface, BB_COLOR, (xlim[0], ylim[0]), (xlim[1], ylim[0]))
            #     pygame.draw.line(bb_surface, BB_COLOR, (xlim[1], ylim[1]), (xlim[0], ylim[1]))
            #     pygame.draw.line(bb_surface, BB_COLOR, (xlim[1], ylim[1]), (xlim[1], ylim[0]))

            #     depth_crop = depth_data[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            #     crop_max_ix = np.argmax(depth_crop, 1)
            #     crop_max_yix = np.argmax(depth_crop[list(range(depth_crop.shape[0])), crop_max_ix])
            #     pygame.draw.circle(bb_surface, BB_COLOR, (xlim[0] + crop_max_ix[crop_max_yix], ylim[0] + crop_max_yix), 10)

            #     # check vals
            #     max_dep = depth_crop[crop_max_yix, crop_max_ix[crop_max_yix]]
            #     min_dep = min([bbox[i, 2] for i in range(8)])
            #     assert(np.max(depth_crop) == max_dep)
            #     print('GOOD' if min_dep < max_dep else 'BAD', min_dep, max_dep)

        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_global_bbox(vehicles):
        bboxes = []
        for vehicle in vehicles:
            bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
            world_cord = ClientSideBoundingBoxes._vehicle_to_world(bb_cords, vehicle)
            bboxes.append(world_cord)
        return bboxes

    @staticmethod
    def get_camera_boxes(bboxes, camera, ret_ixes=False):
        currents = []
        for bbox in bboxes:
            currents.append(ClientSideBoundingBoxes.quick_to_camera(bbox, camera))
        newcurrents = [bb for bb in currents if all(bb[:, 2] > 0)]
        if ret_ixes:
            return newcurrents, [bbi for bbi,bb in enumerate(currents) if all(bb[:, 2] > 0)]
        return newcurrents

    @staticmethod
    def quick_to_camera(world_cord, camera):
        cords_x_y_z = ClientSideBoundingBoxes._world_to_sensor(world_cord, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(bb_cords, vehicle)
        return ClientSideBoundingBoxes.quick_to_camera(world_cord, camera)

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def plot_box(box, lw, color='g', alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array([
        box[:2],
        box[:2] + l/2.*np.array([np.cos(h), np.sin(h)]),
    ])

    plt.fill(simple_box[:, 0], simple_box[:, 1], color=color, edgecolor='k',
             alpha=alpha, zorder=3, linewidth=1.0)
    if not no_heading:
        plt.plot(arrow[:, 0], arrow[:, 1], 'b', alpha=0.5)


def plot_car(state, color='b', alpha=0.5, no_heading=False):
    x, y, h, l, w = state
    plot_box(np.array([x, y, np.cos(h), np.sin(h)]), [l, w],
             color=color, alpha=alpha, no_heading=no_heading)


"""
CAMORDER = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT'
]
"""
CAMORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
