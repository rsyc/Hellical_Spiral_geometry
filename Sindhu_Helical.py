# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:28:35 2025

@author: Mahmood N
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
import sympy
from sympy import symbols, sin, pi
tt = sympy.symbols('tt')  # Define the variable
RR0, ddr, Ht, pich = sympy.symbols('RR0 ddr Ht pich')  # Define constants


depth = 9 # penetration depth or endometrium thickness in mm 
#w = 2*np.pi*80/60  # mean heart rate for non-pregnant women 21-37 years old
vessel_r = 500e-3/2 # vessel lumen radius in mm
pitch = 1000e-3# 116e-3 # pitch height in mm
pitch_star = pitch/(2*vessel_r)
print('p*=', pitch_star)
NLoop = depth/pitch # number of loops 
R_initial = 194e-3 # mm
R_H_star = R_initial/(2*vessel_r)
print('H*=', R_H_star)
R_Final = 0 # Final radius of spiral
b = (R_Final-R_initial)/(2*NLoop*np.pi)
delta_r = 2*NLoop*np.pi*b


# symbolic helpers used in following function
# symbolic helpers used in following function
fff = Ht - pich*tt/(2*pi)
derivative_ff = sympy.diff(fff, tt)
fhelic = (RR0)*sin(tt)
derivative_fhelic = sympy.diff(fhelic, tt)
#print(derivative_f)
def helical_coordinates_2D(t, R0, Height, h_pitch, vessel_r):
    # Compute tangent vectors along centerline | compute derivatives (symbolic -> numeric)
    dx = derivative_fhelic.subs({tt: t, RR0: R0, pi:np.pi})
    dy = derivative_ff.subs({tt: t, Ht: Height, pich: h_pitch, pi:np.pi})
    #print(np.sqrt(float(dx)**2))
    # Normalize tangent vectors
    tangent_norm = np.sqrt(float(dx)**2 + float(dy)**2)
    tx = float(dx) / tangent_norm
    ty = float(dy) / tangent_norm
    # Compute normal vectors (perpendicular to tangent)
    nx = -ty
    ny = tx
    
    x = (R0)*np.sin(t)
    x_upper = (R0)*np.sin(t) + vessel_r*nx
    #print((R0 - dr*t/(2*np.pi))*np.sin(t) + 0.5*derivative_f.subs({tt: t, RR0: R0, ddr: dr, pi:np.pi}))
    x_lower = (R0)*np.sin(t) - vessel_r*nx
    y = Height - h_pitch*t/(2*np.pi)
    y_upper = Height - h_pitch*t/(2*np.pi) + vessel_r*ny
    #print( Height - h_pitch*t/(2*np.pi) + 0.5*derivative_ff.subs({tt: t, Ht: Height, pich: h_pitch, pi:np.pi}))
    y_lower = Height - h_pitch*t/(2*np.pi) - vessel_r*ny
    z = 0.005
    return [x,x_upper,x_lower,y,y_upper,y_lower,z]

# --------- sample points ----------
t_values = np.linspace(0, 2*NLoop*np.pi, 10000)   # denser sampling for smoother STL
points = []
points_upper = []
points_lower = []
x_data = []
x_upper_data = []
x_lower_data = []
y_data = []
y_upper_data = []
y_lower_data = []
z_data = []
for t in t_values:
    #[x,y,z] = spiral_coordinates(t, R_initial,delta_r ,depth, pitch)
    #[x,y,z] = helical_coordinates(t, R_initial ,depth, pitch)
    #[x,x_upper,x_lower,y,y_upper,y_lower,z] = spiral_coordinates_2D(t, R_initial,delta_r ,depth, pitch, vessel_r)
    [x,x_upper,x_lower,y,y_upper,y_lower,z] = helical_coordinates_2D(t, R_initial ,depth, pitch, vessel_r)
    points.append([x,y,z])
    points_upper.append([x_upper,y_upper,z])
    points_lower.append([x_lower,y_lower,z])
    x_data.append(x)
    x_upper_data.append(x_upper)
    x_lower_data.append(x_lower)
    y_data.append(y)
    y_upper_data.append(y_upper)
    y_lower_data.append(y_lower)
    z_data.append(z)
    
    
    
# Saving geometry into .stl format    
# ---------------------------
# Helper: triangle normal
# ---------------------------
def triangle_normal(v1, v2, v3):
    u = v2 - v1
    v = v3 - v1
    n = np.cross(u, v)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return n / norm

# ---------------------------
# Build watertight thin solid and write ASCII STL
# ---------------------------
def points_to_stl(upper_pts, lower_pts, thickness=0.001, filename='spiral_ribbon.stl', close_ends=True):
    """
    upper_pts, lower_pts: lists/arrays of shape (N,3) OR (N,2) (z assumed 0)
    thickness: full thickness in same units (will extrude +/- thickness/2 in z)
    """
    U = np.asarray(upper_pts, dtype=float)
    L = np.asarray(lower_pts, dtype=float)
    if U.shape[1] == 2:
        U = np.column_stack([U, np.zeros(len(U))])
    if L.shape[1] == 2:
        L = np.column_stack([L, np.zeros(len(L))])
    assert U.shape == L.shape, "upper and lower must have same shape"
    n = len(U)
    # ensure same parametric direction
    d_same = np.linalg.norm(U[0]-L[0]) + np.linalg.norm(U[-1]-L[-1])
    d_rev = np.linalg.norm(U[0]-L[-1]) + np.linalg.norm(U[-1]-L[0])
    if d_rev < d_same:
        L = L[::-1]

    half = thickness / 2.0
    top_u = np.column_stack([U[:,0], U[:,1], np.full(n, half)])
    top_l = np.column_stack([L[:,0], L[:,1], np.full(n, half)])
    bot_u = np.column_stack([U[:,0], U[:,1], np.full(n, -half)])
    bot_l = np.column_stack([L[:,0], L[:,1], np.full(n, -half)])

    facets = []

    def add_tri(a,b,c):
        nrm = triangle_normal(a,b,c)
        facets.append((nrm, a.copy(), b.copy(), c.copy()))

    # Create top and bottom faces, and walls
    for i in range(n-1):
        # top face (two triangles)
        add_tri(top_u[i], top_u[i+1], top_l[i])
        add_tri(top_u[i+1], top_l[i+1], top_l[i])
        # bottom face
        add_tri(bot_u[i], bot_l[i], bot_u[i+1])
        add_tri(bot_u[i+1], bot_l[i], bot_l[i+1])
        # upper side wall
        add_tri(top_u[i], bot_u[i], bot_u[i+1])
        add_tri(top_u[i], bot_u[i+1], top_u[i+1])
        # lower side wall
        add_tri(top_l[i], top_l[i+1], bot_l[i])
        add_tri(top_l[i+1], bot_l[i+1], bot_l[i])

    if close_ends:
        # start cap
        add_tri(top_u[0], top_l[0], bot_u[0])
        add_tri(bot_u[0], top_l[0], bot_l[0])
        # end cap
        add_tri(top_u[-1], bot_u[-1], top_l[-1])
        add_tri(bot_u[-1], bot_l[-1], top_l[-1])

    # write ASCII STL
    with open(filename, 'w') as f:
        f.write('solid spiral_ribbon\n')
        for nrm, v1, v2, v3 in facets:
            f.write('  facet normal {: .6e} {: .6e} {: .6e}\n'.format(nrm[0], nrm[1], nrm[2]))
            f.write('    outer loop\n')
            for v in (v1, v2, v3):
                f.write('      vertex {: .6e} {: .6e} {: .6e}\n'.format(v[0], v[1], v[2]))
            f.write('    endloop\n')
            f.write('  endfacet\n')
        f.write('endsolid spiral_ribbon\n')
    return filename

# ---- call exporters ----
upper_pts = np.column_stack([x_upper_data, y_upper_data, z_data])
lower_pts = np.column_stack([x_lower_data, y_lower_data, z_data])
center_pts = np.column_stack([x_data, y_data, z_data])

# scale/convert units if you want (the points are in meters currently)
stl_file = points_to_stl(upper_pts, lower_pts, thickness=0.000, filename='helical_ribbon2D_10000pt_D500_P1000.stl', close_ends=True)

print("Wrote:", stl_file)

# optional quick 2D plot
plt.figure(figsize=(6,8))
plt.plot(x_lower_data, y_lower_data, 'b-')#, label='lower')
plt.plot(x_upper_data, y_upper_data, 'g-')#, label='upper')
plt.plot(x_data, y_data, 'r-')#, label='center')
plt.show()