"""
    Contains miscellanious utility functions
"""

import numpy as np
import torch
import os


def pyt2np(tensor):
    """Transforms a pytorch variable/tensor to numpy tensor"""
    return tensor.cpu().detach().numpy()

def np2pyt(tensor):
    """Transforms a numpy tensor to pytorch"""
    return torch.from_numpy(tensor).float()

def create_exp_folder(parent_dir):
    """Creates an experiment folder with a unique experiment ID in the folder
    specified by parent_dir.

    Parameters
    ----------
    parent_dir: str
        The parent directory in which to save the experiment folder

    Returns
    -------
    exp_dir: str
        The path to the newly create experiment directory
    """
    exp_nr = 1
    exp_dir = parent_dir + ('/exp_%03d' % exp_nr)
    while os.path.exists(exp_dir):
        exp_nr += 1
        exp_dir = parent_dir + ('/exp_%03d' % exp_nr)

    os.makedirs(exp_dir)
    return exp_dir

