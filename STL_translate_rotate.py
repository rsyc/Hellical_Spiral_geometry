# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 08:43:46 2025

@author: rojan
"""
from stl import mesh
import numpy as np
import math


# === Input parameters ===
input_stl = 'C:/Users/rojan/Documents/FSU/Designs/Sindhu/helical_ribbon2D_10000pt_D265_P500 2.stl'       # your original STL file
output_stl = 'C:/Users/rojan/Documents/FSU/Designs/Sindhu/helical_ribbon2D_10000pt_D265_P500 2_translated_rotated.stl' # output STL file
translation_vector = np.array([18.09887, 28.59124, 6.00000])  # (dx, dy, dz) in same units as STL

# Rotation settings
axis = 'z'     # choose 'x', 'y', or 'z'
angle_deg = 90 # rotation angle (anticlockwise, in degrees)

# === Load STL ===
model = mesh.Mesh.from_file(input_stl)

# === Build rotation matrix ===
theta = math.radians(angle_deg)
if axis == 'x':
    R = np.array([[1, 0, 0],
                  [0, math.cos(theta), -math.sin(theta)],
                  [0, math.sin(theta),  math.cos(theta)]])
elif axis == 'y':
    R = np.array([[ math.cos(theta), 0, math.sin(theta)],
                  [0, 1, 0],
                  [-math.sin(theta), 0, math.cos(theta)]])
elif axis == 'z':
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta),  math.cos(theta), 0],
                  [0, 0, 1]])
else:
    raise ValueError("Axis must be 'x', 'y', or 'z'.")

# === Apply rotation and translation ===
# Rotate all triangle vertices
model.vectors = np.dot(model.vectors, R.T)

# === Apply translation ===
model.vectors += translation_vector  # shift all triangles by the translation vector

# === Save new STL ===
model.save(output_stl)

print(f"Model rotated {angle_deg}° anticlockwise around {axis}-axis and translated by {translation_vector}")
print(f"Saved as '{output_stl}'")