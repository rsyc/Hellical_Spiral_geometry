# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:22:43 2025

@author: rojan
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
 
# Finding the Dean number for various radius and length of SAs
# Dean number = \sqrt(2*a/R)*(p*G*a^3/u^2), where 
# a = lumen rsdius
# R = Radius of curvature of the tube. In Zamir's paper it is defined based on the
    # penetration depth of the artery or the endometrium length (H) as H/3
# p = blood density
# G = Pressure  = dp/l
# u = blood viscosity

dp = 80 * 133.322 # pressure change : mmHg to Pa
u = 0.003 # Viscosity : Pa.s
p = 1056 # Density : Kg/m^3
Q = 45*10**(-6)/60 # uterine flow in m^3/s(changed from ml/min to m^3/s)
Q = Q/100          # to find the SA flow we divide the uterine flow by the number of SA ~100
depth = 10*(10**(-3)) # penetration depth or endometrium thickness in mm (unit changed to m)
w = 2*np.pi*80/60  # mean heart rate for non-pregnant women 21-37 years old
NLoop = 5 # number of loops
pitch = depth/NLoop # pitch height in mm 
R_initial = depth/3
R_Final = 0 # Final radius of spiral
b = (R_Final-R_initial)/(2*NLoop*np.pi)
delta_r = 2*NLoop*np.pi*b

def integrand(t, R0, dr, Height, h_pitch):
    x = (R0 - dr*t/(2*np.pi))*np.sin(t)
    y = (R0 - dr*t/(2*np.pi))*np.cos(t)
    z = Height - h_pitch*t/(2*np.pi)
    return np.sqrt(x**2 + y**2 + z**2)

def integrand_radius(t, R0, dr):
    x = (R0 - dr*t/(2*np.pi))*np.sin(t)
    y = (R0 - dr*t/(2*np.pi))*np.cos(t)
    return np.sqrt(x**2 + y**2 )

def Height_values(t, Height, h_pitch):
    return Height - h_pitch*t/(2*np.pi)

# DEAN Number
a_H_Dean =  []#[[[0 for _ in range(3)] for _ in range(11)] for _ in range(10)] # [[[0]*3]*11]*10
count = -1
for radius in range(25, 275, 25):  # vessel radius range in um
    a = radius*(10**(-6)) # unit change from um to m
    count +=1
    
    t_values = np.linspace(0, 2*NLoop*np.pi, 50)
    H_values = Height_values(t_values, depth, pitch)
    # Perform numerical integration over the specified range for various upper limits
    # to get a "cumulative integral" plot
    integral_results = []
    for upper_limit in t_values:
        result, _ = integrate.quad(integrand, 0, upper_limit, \
                                   args=(R_initial,delta_r ,depth, pitch) )  # Integrate from 0 to current upper_limit
        integral_results.append(result)
    integral_radius_results = []
    for upper_limit in t_values:
        result, _ = integrate.quad(integrand_radius, 0, upper_limit, \
                                   args=(R_initial,delta_r) )  # Integrate from 0 to current upper_limit
        integral_radius_results.append(result)
    G = np.divide(dp, integral_results)
    Dean = np.multiply(np.divide(np.sqrt(2*a)*(p*a**3/(u**2)), np.sqrt(integral_radius_results)),G)
    a_H_Dean.append( [a, list(H_values), list(Dean)])
        
        
        

    
# Reynolds Number    
Reynolds =  [[0 for _ in range(2)] for _ in range(10)] # [[[0]*2]*9]
count = -1
for radius in range(25, 275, 25):# vessel radius range in um
    a = radius*(10**(-6)) # unit change from um to m
    count +=1
    
    #for depth in range(10,16):
    Reynolds[count]= [a, p*Q*(2*a)/(u*np.pi*a**2)]   
 
# Womersley Number    
Womersley =  [[0 for _ in range(2)] for _ in range(10)] # [[[0]*2]*9]
count = -1
for radius in range(25, 275, 25):# vessel radius range in um
    a = radius*(10**(-6)) # unit change from um to m
    count +=1
    
    #for depth in range(10,16):
    Womersley[count]= [a, a*np.sqrt(w*p/u)]  


'''
# Dean number based on pitch radius for a spiral tube:
a_H_Dean2 =  [[[0 for _ in range(3)] for _ in np.arange(0, depth+1, pitch/10)] for _ in range(25, 250, 25)] # [[[0]*3]*11]*10
count = -1
for radius in range(25, 250, 25):  # vessel radius range in um
    a = radius*(10**(-6)) # unit change from um to m
    count +=1
    Re = p*Q*(2*a)/(u*np.pi*a**2)
    #for depth in range(10,16):
    #R = depth/3*(10**(-3)) # Starting radius for helical and spiral tubes (Based on Zamir's paper=H/3); unit change from cm to m
    count2 = -1
    for hstep in np.arange(0, depth+1, pitch/10):
        count2 +=1
        R = hstep/3*(10**(-3))        
        Dean_spiral = Re*np.sqrt(R*a*(2*np.pi)**2/((2*np.pi*R)**2+(pitch*10**(-3))**2))#Re*np.sqrt(gamma/(1+Betta**2))
        a_H_Dean2[count][count2]= [a, R, Dean_spiral]

a_H_Dean3 =  [[0 for _ in range(2)] for _ in range(9)] # [[[0]*2]*9]
count = -1
for radius in range(25, 250, 25):  # vessel radius range in um
    a = radius*(10**(-6)) # unit change from um to m
    count +=1
    Re = p*Q*(2*a)/(u*np.pi*a**2)
    #for depth in range(10,16):
    R = depth/3*(10**(-3)) # Starting radius for helical and spiral tubes (Based on Zamir's paper=H/3); unit change from cm to m
    Betta = pitch*10**(-3)/(2*np.pi*R) # pitch unit change to m
    gamma = a/R
    Dean_helical = Re*np.sqrt(gamma/(1+Betta**2))
    a_H_Dean3[count]= [a, R, Dean_helical]
'''        
   

#----------------------------------------------------------------------------
#                                 plots
#----------------------------------------------------------------------------     

'''
x = []
y = []
for i in range(len(a_H_Dean[0])-1, 0, -1):
    x.append(a_H_Dean2[0][i][1]) 
    y.append(a_H_Dean2[0][i][2])
    
plt.plot(x, y) #plt.plot(range(0,len(x)), y)
plt.xlabel("Radius of Curvature (m)")
plt.ylabel("Dean value")
plt.title("Plot of Dean vs curvature radius for vessle diameter of 50 $\mu m$")
plt.xticks(rotation=90)
plt.show() # Displays the plot
'''

#x = []
#y = []
#for i in range(len(a_H_Dean[0])-1, 0, -1):
#    x.append(a_H_Dean[0][i][1]) 
#    y.append(a_H_Dean[0][i][2])

# Dean plot for r=225 um
x = a_H_Dean[1][1][:]   
y = a_H_Dean[1][2][:]
plt.plot(x, y) #plt.plot(range(0,len(x)), y)
plt.xlabel("Vessel Length (m)")#"Radius of Curvature (m)")
plt.ylabel("Dean value")
plt.title("Plot of Dean vs curvature radius for vessle diameter of 50 $\mu m$")
plt.xticks(rotation=90)
plt.show() # Displays the plot
    
    
x = []
y = []
for i in range(0, len(Reynolds)):
    x.append(np.multiply(Reynolds[i][0],10**6)) 
    y.append(Reynolds[i][1])
    
plt.plot(x, y) #plt.plot(range(0,len(x)), y)
plt.xlabel("Radius of vessel (um)")
plt.ylabel("Reynolds Number")
plt.title("Plot of Reynolds vs vessle radius")
plt.xticks(rotation=90)
plt.show() # Displays the plot    


x = []
y = []
for i in range(0, len(Womersley)):
    x.append(np.multiply(Womersley[i][0],10**6)) 
    y.append(Womersley[i][1])
    
plt.plot(x, y) #plt.plot(range(0,len(x)), y)
plt.xlabel("Radius of vessel (um)")
plt.ylabel("Womersley Number")
plt.title("Plot of Womersley vs vessle radius")
plt.xticks(rotation=90)
plt.show() # Displays the plot   