# coding: utf-8
from os import path as osp
import os
root = osp.dirname(osp.abspath(__file__))

# # TestA
# TRAIN_A_ROOT = os.path.join(root, 'TrainA')
# TEST_A_ROOT = os.path.join(root, 'TestA')
# TEST_B_ROOT = os.path.join(root, 'nature')

# O-Haze
OHAZE_ROOT = osp.abspath(osp.join(root, '../data', 'O-Haze'))

# HazeRD
# HazeRD_ROOT = osp.abspath(osp.join(root, '../data', 'HazeRD'))
HazeRD_ROOT = osp.abspath(osp.join(root, '../data'))

# RESIDE
TRAIN_ITS_ROOT = osp.abspath(osp.join(root, '../data', 'ITS'))  # ITS
# TEST_SOTS_ROOT = osp.abspath(osp.join(root, '../data', 'SOTS', 'nyuhaze500'))  # SOTS indoor
TEST_SOTS_ROOT = os.path.join(root, '../data','SOTS', 'outdoor')  # SOTS outdoor
TEST_HSTS_ROOT = os.path.join(root, 'HSTS', 'synthetic')  # HSTS

