#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:46:48 2018

@author: samla
"""

import numpy as np

def mock_x():
    return float(np.random.choice(range(-2, 3)))
def mock_y():
    return float(np.random.choice(range(-1, 2)))
def mock_z():
    return float(np.random.choice(range(8, 11)))

def mock_accel(x, y, z):
    return tuple([x, y, z])

