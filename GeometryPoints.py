# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 08:48:50 2025

@author: rojan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
import sympy
from sympy import symbols, sin, pi
tt = sympy.symbols('tt')  # Define the variable
RR0, ddr, Ht, pich = sympy.symbols('RR0 ddr Ht pich')  # Define constants
%matplotlib qt  


#get_ipython().run_line_magic('matplotlib', 'qt')

# === physical constants / params ===
#dp = 80 * 133.322 # pressure change : mmHg to Pa
#u = 0.003 # Viscosity : Pa.s
#p = 1056 # Density : Kg/m^3
#Q = 45*10**(-6)/60 # uterine flow in m^3/s(changed from ml/min to m^3/s)
#Q = Q/100          # to find the SA flow we divide the uterine flow by the number of SA ~100
depth = 9 # penetration depth or endometrium thickness in mm 
#w = 2*np.pi*80/60  # mean heart rate for non-pregnant women 21-37 years old
pitch = 500e-3# 116e-3 # pitch height in mm
NLoop = depth/pitch # number of loops 
R_initial = 194e-3 # mm
R_Final = 0 # Final radius of spiral
b = (R_Final-R_initial)/(2*NLoop*np.pi)
delta_r = 2*NLoop*np.pi*b
vessel_r = 25e-3/2 # vessel lumen radius in mm

# ---------------------------------------
def spiral_coordinates(t, R0, dr, Height, h_pitch):
    x = (R0 - dr*t/(2*np.pi))*np.sin(t)
    y = (R0 - dr*t/(2*np.pi))*np.cos(t)
    z = Height - h_pitch*t/(2*np.pi)
    return [x,y,z]

def helical_coordinates(t, R0, Height, h_pitch):
    x = (R0)*np.sin(t)
    y = (R0)*np.cos(t)
    z = Height - h_pitch*t/(2*np.pi)
    return [x,y,z] #np.sqrt(x**2 + y**2 )


# symbolic helpers used in following function
ff = (RR0 - ddr*tt/(2*pi))*sin(tt)
derivative_f = sympy.diff(ff, tt)
fff = Ht - pich*tt/(2*pi)
derivative_ff = sympy.diff(fff, tt)
#print(derivative_f)

def spiral_coordinates_2D(t, R0, dr, Height, h_pitch, vessel_r):
    # Compute tangent vectors along centerline | compute derivatives (symbolic -> numeric)
    dx = derivative_f.subs({tt: t, RR0: R0, ddr: dr, pi:np.pi})
    dy = derivative_ff.subs({tt: t, Ht: Height, pich: h_pitch, pi:np.pi})
    #print(np.sqrt(float(dx)**2))
    # Normalize tangent vectors
    tangent_norm = np.sqrt(float(dx)**2 + float(dy)**2)
    tx = float(dx) / tangent_norm
    ty = float(dy) / tangent_norm
    # Compute normal vectors (perpendicular to tangent)
    nx = -ty
    ny = tx
    
    x = (R0 - dr*t/(2*np.pi))*np.sin(t)
    x_upper = (R0 - dr*t/(2*np.pi))*np.sin(t) + vessel_r*nx
    #print((R0 - dr*t/(2*np.pi))*np.sin(t) + 0.5*derivative_f.subs({tt: t, RR0: R0, ddr: dr, pi:np.pi}))
    x_lower = (R0 - dr*t/(2*np.pi))*np.sin(t) - vessel_r*nx
    y = Height - h_pitch*t/(2*np.pi)
    y_upper = Height - h_pitch*t/(2*np.pi) + vessel_r*ny
    #print( Height - h_pitch*t/(2*np.pi) + 0.5*derivative_ff.subs({tt: t, Ht: Height, pich: h_pitch, pi:np.pi}))
    y_lower = Height - h_pitch*t/(2*np.pi) - vessel_r*ny
    z = 0.0
    return [x,x_upper,x_lower,y,y_upper,y_lower,z]


# symbolic helpers used in following function
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
    
# creating figure
#fig = plt.figure()
#ax = Axes3D(fig)

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

# ---------------------------
# Write simple 2D SVG plan view for quick geometry check
# ---------------------------
def write_plan_svg(center_pts, upper_pts, lower_pts, filename='spiral_ribbon_plan.svg', stroke_width=1):
    C = np.asarray(center_pts)[:, :2]
    U = np.asarray(upper_pts)[:, :2]
    L = np.asarray(lower_pts)[:, :2]
    # compute bounds
    all_pts = np.vstack([C,U,L])
    minx, miny = all_pts.min(axis=0) - 0.01
    maxx, maxy = all_pts.max(axis=0) + 0.01
    width = maxx - minx
    height = maxy - miny
    # Create simple SVG with coordinate transform (flip Y for screen)
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" standalone="no"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.6f} {height:.6f}" ')
        f.write(f'preserveAspectRatio="xMidYMid meet">\n')
        # transform to place minx,miny at (0,0)
        # we flip Y so SVG's origin at top-left becomes bottom-left: use transform
        f.write(f'<g transform="translate({{-{minx}}}, {height + miny}) scale(1, -1)">\\n')
        # draw lower (red)
        def polyline_string(pts):
            return ' '.join([f'{p[0]:.6e},{p[1]:.6e}' for p in pts])
        f.write(f'<polyline points="{polyline_string(L)}" stroke="red" fill="none" stroke-width="{stroke_width}"/>\n')
        f.write(f'<polyline points="{polyline_string(C)}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n')
        f.write(f'<polyline points="{polyline_string(U)}" stroke="green" fill="none" stroke-width="{stroke_width}"/>\n')
        f.write('</g>\n')
        f.write('</svg>\n')
    return filename

# ---- call exporters ----
upper_pts = np.column_stack([x_upper_data, y_upper_data, z_data])
lower_pts = np.column_stack([x_lower_data, y_lower_data, z_data])
center_pts = np.column_stack([x_data, y_data, z_data])

# scale/convert units if you want (the points are in meters currently)
stl_file = points_to_stl(upper_pts, lower_pts, thickness=0.000, filename='helical_ribbon2D_1000pt.stl', close_ends=True)
svg_file = write_plan_svg(center_pts, upper_pts, lower_pts, filename='helical_ribbon_plan2D_1000pt.svg')

print("Wrote:", stl_file)
print("Wrote:", svg_file)



'''
## 2D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
# Plot the points
ax.scatter(x_data, y_data, c='blue', marker='o')
# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# Display the plot
plt.show()
'''

## ======3D plot======

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data, c='blue', marker='o')
#ax.scatter(x_upper_data, y_upper_data, z_data, c='green', marker='o')
#ax.scatter(x_lower_data, y_lower_data, z_data, c='red', marker='o')
ax.view_init(90,-90)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Point Plot')
plt.show()


# optional quick 2D plot
plt.figure(figsize=(6,8))
plt.plot(x_lower_data, y_lower_data, 'b-')#, label='lower')
plt.plot(x_upper_data, y_upper_data, 'g-')#, label='upper')
plt.plot(x_data, y_data, 'r-')#, label='center')
#plt.gca().set_aspect('equal', adjustable='box')
#plt.legend()
#plt.title('Spiral ribbon plan view')
plt.show()



## =====Save points====
file_path = "2DhelicalPoints_Ribbon_Lower.dat"
with open(file_path, 'w') as f:
    for p in points_lower:
        #f.write(f"{p[0]} {p[1]}\n") # 2D points
        f.write(f"{p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")    #f.write(f"{p[0]} {p[1]} {p[2]}\n") # 3D points
        

