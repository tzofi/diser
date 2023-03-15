import numpy as np
import open3d as o3d
# import plotly.graph_objects as go
import copy
import os 

def xz_to_xyz(x, z, car_height=0, return_meshcomp=False):

  in_mesh = o3d.io.read_triangle_mesh("tesla_mesh.obj")
  minx,miny,minz = (x_var for x_var in in_mesh.get_min_bound())
  maxx,maxy,maxz = (x_var for x_var in in_mesh.get_max_bound())
  cx, cy, cz = [(minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2]
  mesh_s = copy.deepcopy(in_mesh)
  # mesh_s = mesh_s.translate(-1 * np.array([cx, cy, cz]))
  if car_height != 0:
    mesh_s = mesh_s.scale(car_height / (maxy - miny), center=[0,0,0])
  mesh = mesh_s
  
  if mesh.is_empty(): exit()
  cube = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  # Create a scene and add the triangle mesh
  scene = o3d.t.geometry.RaycastingScene()
  cube_id = scene.add_triangles(cube)
  y_or = 50
  rays = o3d.core.Tensor([[-x, y_or, -z, 0, -1, 0]],
                        dtype=o3d.core.Dtype.Float32)

  ans = scene.cast_rays(rays)

  ret_y = (y_or-ans['t_hit'].numpy())[0]

  if np.isinf(ret_y):
    ret_y = 0.744

  if return_meshcomp:
    return(x, ret_y, z, mesh)
  else:
    return(x, ret_y, z)

  

