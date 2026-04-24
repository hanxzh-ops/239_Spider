#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:29:00 2025

@author: jamesz
"""

# src/robot.py
import mujoco
import numpy as np

class SpiderRobot:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Map actuator names to IDs for fast access
        # Structure: self.actuators[leg_idx][0=Coxa, 1=Femur, 2=Tibia]
        self.actuators = np.zeros((6, 3), dtype=int)
        
        for i in range(6):
            # Using your XML naming convention "act_legX_part"
            self.actuators[i, 0] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_leg{i}_coxa")
            self.actuators[i, 1] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_leg{i}_femur")
            self.actuators[i, 2] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_leg{i}_tibia")

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_angles(self, all_leg_angles):
        """
        all_leg_angles: 6x3 numpy array of radians
        """
        for i in range(6):
            for j in range(3):
                act_id = self.actuators[i, j]
                # Send control command
                # Note: MuJoCo handles the clipping to ctrlrange automatically
                self.data.ctrl[act_id] = all_leg_angles[i, j]