#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:43:54 2025

@author: jamesz
"""

import mujoco
import os

xml_path = "assets/hexapod.xml"

model = mujoco.MjModel.from_xml_path(xml_path)

print("\n=== ACTUATORS FOUND IN MODEL ===")
for i in range(model.nu):
    name = model.names[model.name_actuatoradr[i]:].split(b'\x00', 1)[0].decode()
    print(f"{i}: {name}")
