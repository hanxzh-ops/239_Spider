#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:35:52 2025

@author: jamesz
"""

# src/mj_interface.py
"""
MujocoInterface
A lightweight wrapper around MuJoCo for controlling the hexapod.

This provides:
 - model, data
 - step()
 - set_joint_targets(dict)
 - get_joint_angles()
 - get_body_pose()

Used by controller.py and main.py
"""
# src/mj_interface.py

import mujoco
import numpy as np


class MujocoInterface:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # FIX: Correct actuator name parsing
        self.act_map = {}
        for i in range(self.model.nu):
            # Read actuator name from model.names
            addr = self.model.name_actuatoradr[i]
            name = self._read_name(addr)
            self.act_map[name] = i

        # Joint name -> qpos address
        self.jnt_qposadr = {}
        for i in range(self.model.njnt):
            addr = self.model.name_jntadr[i]
            jname = self._read_name(addr)
            adr = self.model.jnt_qposadr[i]
            self.jnt_qposadr[jname] = adr

    def _read_name(self, addr):
        """Read a null-terminated string from model.names buffer."""
        buf = self.model.names[addr:]
        return buf.split(b'\x00', 1)[0].decode()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_joint_targets(self, joint_map):
        for j_name, target in joint_map.items():
            act_name = "act_" + j_name.replace("_j", "")
            if act_name in self.act_map:
                idx = self.act_map[act_name]
                self.data.ctrl[idx] = float(target)
            else:
                print(f"[MujocoInterface] Warning: actuator '{act_name}' not found.")

    def get_joint_angles(self):
        angles = {}
        for j_name, adr in self.jnt_qposadr.items():
            angles[j_name] = float(self.data.qpos[adr])
        return angles

    def get_body_pose(self, body_name="torso"):
        body_id = self.model.body(body_name).id
        pos = np.array(self.data.xpos[body_id])
        quat = np.array(self.data.xquat[body_id])
        return pos, quat
