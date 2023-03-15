## Creates Default Scenes 
import torch
from redner_scene import *

MAX_BOUNCES = 3
NUM_SAMPLES = 1024

def move_camera(res, hfov, new_look_at, new_x, new_y=3.0, new_z=5.0):
    res = (res, res)
    eye_pos = [new_x, new_y, new_z]
    look_at = [new_look_at, 3.0, 0.0] #3.01166],
    up = [0.0, 1.0, 0.0]

    camera = Camera(hfov=torch.tensor([hfov]), res=res)
    camera.set_camera_matrix(eye_pos=eye_pos,
                             lookAtPoint=np.array(look_at), 
                             upGuidance=up)
    camera.camera = camera.camera.to(pyredner.get_device())

    cam = pyredner.Camera(position = torch.tensor(eye_pos),
                          look_at = torch.tensor(look_at),
                          up = torch.tensor(up),
                          fov = torch.tensor([hfov]), # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = res,
                          camera_type = redner.CameraType.perspective)
    return cam, camera


def get_obs_v3(shapes, cam, camera, area_light, materials, res):
    if type(area_light) != np.ndarray:
        area_light = [area_light]
    # Setup the scene. We don't need lights.
    scene = pyredner.Scene(camera = cam,
                           shapes = shapes,
                           materials = materials,
                           area_lights = area_light)

    # We output the shape id, so that we can shape it later
    args = pyredner.RenderFunction.serialize_scene(\
        # ambient_light=[0.2, 0.2, 0.2],
        scene = scene,
        num_samples = NUM_SAMPLES,
        # Set max bounces to 0, we don't need lighting.
        max_bounces = MAX_BOUNCES,
        #channels = [redner.channels.depth])
        channels = [redner.channels.radiance, redner.channels.depth])
        # Use the diffuse color as the output
        #channels = [])
    render = pyredner.RenderFunction.apply
    #sys.stdout = old_stdout # reset old stdout
    
    img = render(0, *args)
    image = img[:, :, :3]
    image = torch.clip(image, 0., 1.0)
    depth = img[:, :, 3:]
    #depth = img
    depth = get_z_from_range(res, camera, depth.squeeze())
    return image, depth

def make_cuboid(pos, material_id):

    m1, m2, m3 = material_id[0], material_id[1], material_id[2]
    
    wall = torch.tensor(pos,
            device = pyredner.get_device(),
            requires_grad = True)
    vs = MakeBox(wall)
    boxLidBottom_indices = torch.tensor([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4]], dtype = torch.int32, device = pyredner.get_device())
    shape_boxLidBottom = pyredner.Shape(\
        vertices = vs,
        indices = boxLidBottom_indices,
        material_id = m1)
    boxLeftRight_indices = torch.tensor([[0, 1, 5], [5, 4, 0], [6, 2, 3], [3, 7, 6]], dtype = torch.int32, device = pyredner.get_device())
    shape_boxLeftRight = pyredner.Shape(\
        vertices = vs,
        indices = boxLeftRight_indices,
        material_id = m2)
    boxFrontBack_indices = torch.tensor([[1, 2, 5], [2, 6, 5], [3, 0, 4], [4, 7, 3]], dtype = torch.int32, device = pyredner.get_device())
    shape_boxFrontBack = pyredner.Shape(\
        vertices = vs,
        indices = boxFrontBack_indices,
        material_id = m3)

    return shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack

def return_sun_as_light_default(light_pos=np.array([-4.414*50, 7*50, 4.414*50]), shape_id=0, intensity = 150000000., light_size = 0.1, type=None):
    sun_as_cuboid_verts = [[light_pos[0]-light_size, light_pos[1], light_pos[2]-light_size], 
                        [light_pos[0]-light_size, light_pos[1], light_pos[2]+light_size], 
                        [light_pos[0]+light_size, light_pos[1], light_pos[2]+light_size],
                        [light_pos[0]+light_size, light_pos[1], light_pos[2]-light_size]]
    sun_as_cuboid_verts = np.array(sun_as_cuboid_verts, dtype=np.float32)


    sun_as_cuboid_inds = [[0,3,2],[0,1,2]]
    sun_as_cuboid_inds = np.array(sun_as_cuboid_inds, dtype=np.int32)

    shape_light = pyredner.Shape(vertices=torch.tensor(sun_as_cuboid_verts, 
                                                        device=pyredner.get_device()), 
                                indices=torch.tensor(sun_as_cuboid_inds, 
                                                    device=pyredner.get_device()), material_id=0)
    if not isinstance(intensity, list):
        intensity_ = np.array([intensity, intensity, intensity])
    else:
        intensity_ = intensity
    if type is None:
        area_light = pyredner.AreaLight(shape_id=shape_id, intensity=torch.Tensor(intensity_), directly_visible=False)
    if type == "point":
        area_light = pyredner.AreaLight(shape_id=shape_id, intensity=torch.Tensor(intensity_), directly_visible=False)


    return area_light, shape_light

def get_simple_scene1():
    """
    returns: shapes, area light
    """

    target1_pos=[-1.0, -1.0, 2.0, 3.5, 2.0, 45.0]
    target2_pos=[-4.0, 4.0, 1.0, 2.0, 3.0, 20.0]
    """
    Target pos: [x, z, y, width?, depth?, angle]
    Modify this to define your scene
    """
    # LIGHT (these verts should go first in shapes)
    area_light, shape_light = return_sun_as_light_default(intensity = 150000000)
    shapes = [shape_light]

    # WALL 1
    shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid([-10.0, -10.0, 0.0, 5.0, 20.0, 90.0], material_id=[5,5,5])
    shapes.extend([shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack])
    
    # TARGET 1
    shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(target1_pos, material_id=[1,2,0])
    shapes.extend([shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack]) 
    # GROUND PLANE
    ground_indices = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype = torch.int32, device = pyredner.get_device())
    ground_verts = torch.tensor([[-2000000.0, 0.0, -2000000.0], [-2000000.0, 0.0, 2000000.0],
                                [2000000.0, 0.0, 2000000.0], [2000000.0, 0.0, -2000000.0]], 
                                device=pyredner.get_device())

    shape_ground = pyredner.Shape(\
        vertices = ground_verts,
        indices = ground_indices,
        material_id = 3)
    shapes.extend([shape_ground])

    return shapes, area_light

def get_simple_scene_ibo1(box_x=-2.0, box_z=-5.0, shadow_light_pos=np.array([10., 6, -5.])):
    """
    returns: shapes, area light
    """

    target1_pos=[box_x, box_z, 2.0, 3.0, 2.0, 90.0]
    # target2_pos=[-4.0, 4.0, 1.0, 2.0, 3.0, 20.0]
    """
    Target pos: [x, z, y, width?, depth?, angle]
    Modify this to define your scene
    """
    # Global LIGHTing: Controls the global illumination of the scene  
    light_pos = np.array([10., 40, 10.])
    area_light_1, shape_light = return_sun_as_light_default(light_pos=light_pos, intensity = 3000.00, light_size = 1.)
    # other intesities: 3k, 2k, the higher the more the overall scene is visible 

    # Shadow LIGHTing: Is there to introduce lighting in the scene which causes shadows 
    shape_light_2 = None
    # light_pos = np.array([100., 100, 100.])
    area_light_2, shape_light_2 = return_sun_as_light_default(light_pos=shadow_light_pos, shape_id=1, intensity = 4000.0, light_size = 1, type='point')
    # intensity should be higher than the global lighting to enable shadows, if it is too high then it wiill cause a big white spot 
    
    if shape_light_2:
        shapes = [shape_light, shape_light_2]
        area_light = np.array([area_light_1, area_light_2])
    else:
        shapes = [shape_light]
        area_light = area_light_1
        
    # WALL 1
    wall_params = [-3.0, -3.0, 0.0, 10.0, 17.0, 90.0]
    shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(wall_params, material_id=[0,5,5])
    shapes.extend([shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack])
    
    # TARGET 1
    shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack = make_cuboid(target1_pos, material_id=[1,2,2])
    shapes.extend([shape_boxLidBottom, shape_boxLeftRight, shape_boxFrontBack]) 
    # GROUND PLANE
    ground_indices = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype = torch.int32, device = pyredner.get_device())
    ground_verts = torch.tensor([[-2000000.0, 0.0, -2000000.0], [-2000000.0, 0.0, 2000000.0],
                                [2000000.0, 0.0, 2000000.0], [2000000.0, 0.0, -2000000.0]], 
                                device=pyredner.get_device())

    shape_ground = pyredner.Shape(\
        vertices = ground_verts,
        indices = ground_indices,
        material_id = 3)
    shapes.extend([shape_ground])

    return shapes, area_light
