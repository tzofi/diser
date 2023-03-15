import torch
from redner_scene import *
from get_premade_scenes import make_cuboid, return_sun_as_light_default
from utils import get_reward_for_projected_verts, rotate_around_z
import materials_file
MAX_BOUNCES = 0
NUM_SAMPLES = 256

class TriangulationSceneBounded():
    def __init__(self, render_box=True):
        # MAX Bounds for the box 
        self.min_bounds_x = -40.
        self.max_bounds_x = 41.
        self.min_bounds_z = -20.
        self.max_bounds_z = 70.

        self.curr_box_verts = None

        self.light_direction = torch.tensor(0.)

        self.use_box_as_object = render_box

    def set_camera(self, res, hfov, look_at, eye_pos):
        up = [0.0, 1.0, 0.0]

        # check bounds for box  x
        if eye_pos[0] > self.max_bounds_x:
            eye_pos[0] = self.max_bounds_x
        if eye_pos[0] < self.min_bounds_x:
            eye_pos[0] = self.min_bounds_x
        # check bounds for box z
        if eye_pos[2] > self.max_bounds_z:
            eye_pos[2] = self.max_bounds_z
        if eye_pos[2] < self.min_bounds_z:
            eye_pos[2] = self.min_bounds_z

        camera = Camera(hfov=torch.tensor([hfov]), res=res)
        camera.set_camera_matrix(eye_pos=np.array(eye_pos),
                                lookAtPoint=np.array(look_at).astype(float), 
                                upGuidance=up)
        camera.camera = camera.camera.to(pyredner.get_device())
        cam = pyredner.Camera(position = torch.tensor(eye_pos),
                            look_at = torch.tensor(look_at).float(),
                            up = torch.tensor(up),
                            fov = torch.tensor([hfov]), # in degree
                            clip_near = 1e-2, # needs to > 0
                            resolution = res,
                            camera_type = redner.CameraType.perspective)
        return cam, camera

    def set_camera2(self, res, hfov, look_at, eye_pos, vport):
        up = [0.0, 1.0, 0.0]

        # check bounds for box  x
        if eye_pos[0] > self.max_bounds_x:
            eye_pos[0] = self.max_bounds_x
        if eye_pos[0] < self.min_bounds_x:
            eye_pos[0] = self.min_bounds_x
        # check bounds for box z
        if eye_pos[2] > self.max_bounds_z:
            eye_pos[2] = self.max_bounds_z
        if eye_pos[2] < self.min_bounds_z:
            eye_pos[2] = self.min_bounds_z

        camera = Camera(hfov=torch.tensor([hfov]), res=res)
        camera.set_camera_matrix(eye_pos=np.array(eye_pos),
                                lookAtPoint=np.array(look_at).astype(float), 
                                upGuidance=up)
        camera.camera = camera.camera.to(pyredner.get_device())
        cam = pyredner.Camera(position = torch.tensor(eye_pos),
                            look_at = torch.tensor(look_at).float(),
                            up = torch.tensor(up),
                            fov = torch.tensor([hfov]), # in degree
                            clip_near = 1e-2, # needs to > 0
                            resolution = res,
                            viewport = vport,
                            camera_type = redner.CameraType.perspective)
        return cam, camera
        
    def return_bev_camera(self, height = 100.):
        """
        For Debugging purposes, returns a bev of the scene
        """
        res = (200, 200)
        up = [0.0, 1.0, 0.0]

        hfov = 100.
        look_at = [00., 0.0, 0.]
        eye_pos = [0.0, height, 3.0]

        camera = Camera(hfov=torch.tensor([hfov]), res=res)
        camera.set_camera_matrix(eye_pos=eye_pos,
                                lookAtPoint=np.array(look_at), 
                                upGuidance=up)
        camera.camera = camera.camera.to(pyredner.get_device())

        redner_cam = pyredner.Camera(position = torch.tensor(eye_pos),
                            look_at = torch.tensor(look_at),
                            up = torch.tensor(up),
                            fov = torch.tensor([hfov]), # in degree
                            clip_near = 1e-2, # needs to > 0
                            resolution = res,
                            camera_type = redner.CameraType.perspective)
        return redner_cam, camera

    def get_obs_from_agent(self, finder_object, redner_cam, camera, light_obj, res):
        """return observation from an agent, images are gamma corrected and clipped
        Args:
            objects
            cam 
            camera 
            light_obj 
            res 
        Returns:
            torch.tensor, None : image, None
        """
        objects = self.base_objects + [obj for obj in finder_object] + [light_obj]
        # if self.render_box:
        #     obj = self.render_box()
        # else:
        #     obj = self.render_sphere()
        #objects.extend(finder_object)
        #objects.extend([light_obj])
        scene = pyredner.Scene(objects=objects, camera=redner_cam)
        img = pyredner.render_pathtracing(scene, num_samples = NUM_SAMPLES, max_bounces = MAX_BOUNCES, use_primary_edge_sampling=False, use_secondary_edge_sampling=False)
        image = img[:, :, :3]
        image = torch.pow(image, 1.0 / 2.2)  # gamma correction
        image = torch.clip(image, 0., 1.0)
        del scene
        return image # depth = None

    def prepare_base_scene(self, enable_occluder=False, bound_scene=True):
        objects = []
        # Bound the Scene
        if bound_scene: 
            objects = self._populate_scene_with_walls(objects)
        # WALL 1
        if enable_occluder:
            objects = self._populate_scene_with_occluder(objects)

        # GROUND PLANE
        ground_indices = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype = torch.int32, device = pyredner.get_device())
        ground_verts = torch.tensor([[-2000000.0, 0.0, -2000000.0], [-2000000.0, 0.0, 2000000.0],
                                    [2000000.0, 0.0, 2000000.0], [2000000.0, 0.0, -2000000.0]], 
                                    device=pyredner.get_device())
        idx = materials_file.materials_dict["mat_beige"]
        ground_obj = pyredner.Object(vertices = ground_verts, indices = ground_indices, material = idx)
        objects.extend([ground_obj])

        self.base_objects = objects

    def illuminate_scene(self, light_pos, delta_z, intensity = torch.tensor([500000.0, 500000.0, 500000.0])):
        """_summary_
        Args:
            light_pos (tensor): _description_
            delta_z (tensor.float): _description_
        Returns:
            _type_: _description_
        """
        # generate_quad_light outputs a light object 
        look_at_pos = light_pos + torch.tensor([0,0,100]).float().to(light_pos.device)
        light_pos = self.verify_3dpos(light_pos)
        look_at = rotate_around_z(look_at_pos.cpu().numpy(), (self.light_direction + delta_z).cpu().numpy())

        light_obj = pyredner.generate_quad_light(position = light_pos.cuda(),
                                     look_at = torch.tensor(look_at).float().cuda(),
                                     size = torch.tensor([0.1, 0.1]).cuda(),#size = torch.tensor([0.1, 0.1]).cuda(),
                                     intensity = intensity.cuda())

        return light_obj

    def render_sphere(self, position = [5.0, 10.0, 5.0], radius = 2, material_id="m_sphere"):
        """Create Sphere instead of Cuboid as the object.
        Args:
            position (list, optional): Defaults to [5.0, 10.0, 5.0].
            material_id (str, optional):  Defaults to "m_sphere".
        """
        vertices, indices, uvs, normals = pyredner.generate_sphere(theta_steps = 64, phi_steps = 128)
        position = self.verify_3dpos(position)

        vertices = vertices * radius + torch.tensor(position).to(vertices.device)
        m = materials_file.materials_dict[material_id]
        sphere_obj = pyredner.Object(vertices = vertices, indices = indices, uvs = uvs, normals = normals, material = m)

        return [sphere_obj]

    def render_box(self, box_x=-2.0, box_z = -5.0, box_lengths = [5.0, 10.0, 5.0]):
        """
        Target pos: [x, z, y, width?, depth?, angle]
        Modify this to define your scene
        """
        objects = []
        # check bounds for box  x
        box_x, _, box_z = self.verify_3dpos([box_x, 10., box_z])
        
        target1_pos = [box_x, box_z, box_lengths[0], box_lengths[1], box_lengths[2], 135.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(target1_pos, material_id=[1,1,2])
        self.curr_box_verts = [] 
        self.curr_box_verts.extend([shape_boxFrontBack.vertices, shape_boxLeftRight.vertices, shape_boxLidBottom.vertices])

        idx_1 = materials_file.materials_dict["mat_green"]
        idx_2 = materials_file.materials_dict["mat_blue"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx_1)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx_1)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx_2)        
        objects.extend([obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack])

        return objects

    def verify_3dpos(self, light_pos):
        # (v nice pos): light_pos = np.array([-20., 100., 50.]), intensity= 3000000.0, light_size = 0.1
        # check bounds for box  x
        if light_pos[0] > self.max_bounds_x:
            light_pos[0] = self.max_bounds_x
        if light_pos[0] < self.min_bounds_x:
            light_pos[0] = self.min_bounds_x
        # check bounds for box z
        if light_pos[2] > self.max_bounds_z:
            light_pos[2] = self.max_bounds_z
        if light_pos[2] < self.min_bounds_z:
            light_pos[2] = self.min_bounds_z
        return light_pos

    def _populate_scene_with_walls(self, objects):
        # front wall
        wall_params = [-500.0, -30.0, 1.0, 30.0, 1000.0, 90.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[5,5,5])
        idx = materials_file.materials_dict["mat_blue"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx)        
        objects.extend([obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack])

        # left wall
        wall_params = [-50.0, -50.0, 1.0, 30.0, 1000.0, 0.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[5,5,5])
        idx = materials_file.materials_dict["mat_red"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx)
        objects.extend([obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack])

        # back wall
        wall_params = [-500.0, 100.0, 1.0, 30.0, 1000.0, 90.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[5,5,5])
        idx = materials_file.materials_dict["mat_grey"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx)
        objects.extend([obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack])

        # rightw wall
        wall_params = [50.0, -50.0, 1.0, 30.0, 1000.0, 0.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[5,5,5]) #0,0,0
        idx = materials_file.materials_dict["mat_green"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx)
        objects.extend([obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack])

        return objects

    def _populate_scene_with_occluder(self, shapes):
        wall_params = [-3.0, -3.0, 1.0, 100.0, 17000.0, 90.0]
        shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[0,5,5])
        idx_0 = materials_file.materials_dict["mat_red"]
        idx_5 = materials_file.materials_dict["mat_grey"]

        obj_boxLidBottom = pyredner.Object(vertices = shape_boxLidBottom.vertices, indices = shape_boxLidBottom.indices, material = idx_0)
        obj_boxLeftRight = pyredner.Object(vertices = shape_boxLeftRight.vertices, indices = shape_boxLeftRight.indices, material = idx_5)
        obj_boxFrontBack = pyredner.Object(vertices = shape_boxFrontBack.vertices, indices = shape_boxFrontBack.indices, material = idx_5)
        
        occluder_object = [obj_boxLidBottom, obj_boxLeftRight, obj_boxFrontBack]

        return occluder_object
