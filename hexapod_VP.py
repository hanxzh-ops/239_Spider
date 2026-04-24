#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 02:08:12 2025

@author: jamesz
"""
'''
Hexapod robot test program
Adapted from quadruped
'''

__author__ = 'regishsu'

'''
Hexapod robot test program
Adapted from quadruped
'''

from vpython import *
from math import sin, cos, sqrt, acos, atan2

# -------------------------------
# Hexapod body dimensions
# -------------------------------
body_x, body_y, body_z = 71, 71, 27
length_side = body_x

coxa_len, femur_len, tibia_len = 27.5, 70, 80

draw_x_offset = body_x / 2
draw_y_offset = body_y / 2
draw_z_offset = body_z

x_offset = 0
z_ground = -draw_z_offset
z_stand = -50

x_range = (coxa_len + femur_len * 0.55)
y_range = x_range
z_range = z_stand
t_step = 2

y_start = 0
y_step = body_y * 0.6
z_up = -12
x_up_offset = 0

x_current = [0]*6
y_current = [0]*6
z_current = [0]*6

STAY = 255

# placeholders for leg objects
coxa, femur, tibia = [0]*6, [0]*6, [0]*6

# -------------------------------
# VPython scene
# -------------------------------
scene = canvas(title="Hexapod Simulation", width=800, height=600, center=vector(0,0,0))

# Draw global axes
curve(pos=[vector(0,0,0), vector(0,0,250)], color=color.red)
curve(pos=[vector(0,0,0), vector(0,250,0)], color=color.green)
curve(pos=[vector(0,0,0), vector(250,0,0)], color=color.blue)

# -------------------------------
# Utility to create a leg
# -------------------------------
def create_legs(i):
    # coxa
    coxa_obj = cylinder(pos=vector(0,0,0), axis=vector(coxa_len,0,0),
                        radius=6, color=color.red)
    # femur
    femur_obj = cylinder(pos=coxa_obj.pos + coxa_obj.axis, axis=vector(femur_len,0,0),
                         radius=6, color=color.green)
    # tibia
    tibia_obj = cylinder(pos=femur_obj.pos + femur_obj.axis, axis=vector(0,0,-tibia_len),
                         radius=6, color=color.blue)
    return coxa_obj, femur_obj, tibia_obj

# -------------------------------
# Forward kinematics: axis to angle
# -------------------------------
def axis_to_angle(x, y, z):
    w = sqrt(x**2 + y**2) if x>=0 else -sqrt(x**2 + y**2)
    v = w - coxa_len

    alpha_tmp = (femur_len**2 - tibia_len**2 + v**2 + z**2)/(2*femur_len*sqrt(v**2 + z**2))
    alpha_tmp = max(min(alpha_tmp,1),-1)
    alpha = atan2(z,v) + acos(alpha_tmp)

    beta_tmp = (femur_len**2 + tibia_len**2 - v**2 - z**2)/(2*femur_len*tibia_len)
    beta_tmp = max(min(beta_tmp,1),-1)
    beta = acos(beta_tmp)

    gamma = atan2(y,x) if w>=0 else atan2(-y,-x)
    return alpha, beta, gamma

# -------------------------------
# Draw leg with angles
# -------------------------------
def draw_legs(leg, a, b, g):
    # Directions based on leg index
    x_dir = 1 if leg in [2,3,5] else -1
    y_dir = 1 if leg in [0,2,4] else -1
    z_dir = -1

    coxa[leg].axis = vector(x_dir*cos(g)*coxa_len, y_dir*sin(g)*coxa_len, 0)
    femur[leg].axis = vector(femur_len*cos(a), 0, femur_len*sin(a))
    tibia[leg].axis = vector(-tibia_len*cos(b), 0, -tibia_len*sin(b))
    coxa[leg].pos = vector(x_dir*draw_x_offset, y_dir*draw_y_offset, draw_z_offset)

# -------------------------------
# Set leg position
# -------------------------------
def set_legs(leg, x, y, z):
    global x_current, y_current, z_current
    xx = x if x != STAY else x_current[leg]
    yy = y if y != STAY else y_current[leg]
    zz = z if z != STAY else z_current[leg]

    x_current[leg], y_current[leg], z_current[leg] = xx, yy, zz
    a,b,g = axis_to_angle(xx,yy,zz)
    draw_legs(leg,a,b,g)

def wait_all_reach():
    sleep(0.2)

# -------------------------------
# Sit / Stand
# -------------------------------
def sit():
    for leg in range(6):
        set_legs(leg, STAY, STAY, z_ground)

def stand():
    for leg in range(6):
        set_legs(leg, STAY, STAY, z_stand)

# -------------------------------
# Walk functions (tripod gait)
# -------------------------------
def step_forward(step):
    n_step = step
    while n_step > 0:
        n_step -= 1
        # Tripod 1: legs 0,3,4
        for leg in [0,3,4]:
            set_legs(leg, x_range + x_up_offset, y_start, z_up)
        wait_all_reach()
        for leg in [0,3,4]:
            set_legs(leg, x_range + x_up_offset, y_start + t_step*y_step, z_stand)
        wait_all_reach()
        # Tripod 2: legs 1,2,5
        for leg in [1,2,5]:
            set_legs(leg, x_range + x_up_offset, y_start, z_up)
        wait_all_reach()
        for leg in [1,2,5]:
            set_legs(leg, x_range + x_up_offset, y_start + t_step*y_step, z_stand)
        wait_all_reach()

def step_back(step):
    n_step = step
    while n_step > 0:
        n_step -= 1
        # Tripod 1
        for leg in [0,3,4]:
            set_legs(leg, x_range + x_up_offset, y_start, z_up)
        wait_all_reach()
        for leg in [0,3,4]:
            set_legs(leg, x_range + x_up_offset, y_start - t_step*y_step, z_stand)
        wait_all_reach()
        # Tripod 2
        for leg in [1,2,5]:
            set_legs(leg, x_range + x_up_offset, y_start, z_up)
        wait_all_reach()
        for leg in [1,2,5]:
            set_legs(leg, x_range + x_up_offset, y_start - t_step*y_step, z_stand)
        wait_all_reach()

# -------------------------------
# Draw ground
# -------------------------------
z_stand_draw = z_stand - z_ground
for xx in range(-200, 200, 10):
    curve(pos=[vector(xx, -200, z_stand_draw), vector(xx, 200, z_stand_draw)],
          color=color.gray(0.2) if xx % 50 else color.yellow)
for yy in range(-200, 200, 10):
    curve(pos=[vector(-200, yy, z_stand_draw), vector(200, yy, z_stand_draw)],
          color=color.gray(0.2) if yy % 50 else color.yellow)

# -------------------------------
# Create body & legs
# -------------------------------
body = box(pos=vector(0,0,body_z/2), length=body_x, height=body_y,
           width=body_z, color=color.magenta)
for i in range(6):
    coxa[i], femur[i], tibia[i] = create_legs(i)

# -------------------------------
# Main simulation
# -------------------------------
stand()

