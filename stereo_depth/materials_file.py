import pyredner
import torch

materials_dict = {

    "mat_red" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([1.0, 0.0, 0.0],
        device = pyredner.get_device())),

    "mat_green" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([0.0, 1.0, 0.0],
        device = pyredner.get_device())),

    "mat_blue" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([0.0, 0.0, 1.0],
        device = pyredner.get_device())),

    "mat_beige" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([1.0, 0.8, 0.3],
        device = pyredner.get_device())),

    "mat_white" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([1.0, 1.0, 1.0],
        device = pyredner.get_device())),

    "mat_grey" : pyredner.Material(\
        diffuse_reflectance = torch.tensor([0.2, 0.2, 0.2],
        device = pyredner.get_device())),

    "m_sphere" : pyredner.Material(\
        diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), 
        device = pyredner.get_device())),

    "m_specular_low_roughness" : pyredner.Material(\
        specular_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()),
        roughness = torch.tensor([0.001], device = pyredner.get_device())),

    "m_glossy" : pyredner.Material(\
        specular_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()),
        roughness = torch.tensor([0.1], device = pyredner.get_device())),

    "m_plastic" : pyredner.Material(\
        diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()),
        specular_reflectance = torch.tensor((0.2, 0.2, 0.2), device = pyredner.get_device()),
        roughness = torch.tensor([0.001], device = pyredner.get_device())),

}

def find_material_idx(key=None):
    if key == None:
      return list(materials_dict.values())  
    keys = list(materials_dict.keys())
    values = list(materials_dict.values())
    idx = keys.index(key)
    
    return idx, values
