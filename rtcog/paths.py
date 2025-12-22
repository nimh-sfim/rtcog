"""
Directory structure definitions for the rtcog software package.

This module defines key paths used throughout the codebase.
All paths are constructed relative to the location of this file.
"""

import os.path as osp

CODE_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.abspath(osp.join(CODE_DIR, '..'))
RESOURCES_DIR = osp.join(CODE_DIR, 'resources/')
CONFIG_DIR = osp.join(CODE_DIR, "config/")
DATA_DIR = osp.join(ROOT_DIR, "tests/data/")

SIMULATION_DIR = osp.join(ROOT_DIR, 'Simulation')
LAPTOP_DIR = osp.join(SIMULATION_DIR, 'Laptop/')
REALTIME_DIR = osp.join(SIMULATION_DIR, 'Realtime/')
SCANNER_DIR = osp.join(SIMULATION_DIR, 'Scanner/')
OUTPUT_DIR = osp.join(SIMULATION_DIR, 'outputs')
