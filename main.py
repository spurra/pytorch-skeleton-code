"""
    Script to train a baseline model
"""
import numpy as np
import os
from queue import Queue
import time
import shutil
import argparse
import matplotlib
matplotlib.use("Agg")
#import sys
#sys.path.append('./') # Because of the virtualenv in the servers
# Own packages
# Import this first, otherwise it will crash. (Loads first TF then torch)
from src.util.log_interface import LogInterface
from src.train.trainer import Trainer
import src.models.inception as inception
from src.data.data_handler import DataQueue
from src.util.misc import pyt2np, np2pyt, create_exp_folder, make_folder
from src.util.visualization import draw_keypoints
from src.util.img_transformations import rotate, crop_hand, flipLR, resize

import torch
import torch.nn as nn
import torch.optim as optim



# TODO
# Replace tensorboard with Dash
# A logger class that is multithreaded
# Include the sacred framework

if __name__ == '__main__':

    """ Options
        -------
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument("-d", "--checkpoint_path", type=str, 
            help='''Path to directory containing model weights and optimizer 
            state of a previous training run'''
            )
    args = parser.parse_args()
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = None
    # Number of epochs to train
    n_epochs = 1
    # Frequency of checkpoint saving
    save_frequency = 5
    # Path to the input images and corresponding 2D keypoints
    ds_train_path = 'data/processed/training/training_set.h5'
    # Path to the test images and corresponding 2D keypoints
    ds_test_path = 'data/processed/test/test_set.h5'
    # Path to the output folder
    output_path = 'out'
    # Number of batches in the queue
    queue_size = 64
    n_workers = 1
    learning_rate = 1e-4
    batch_size = 32
    # Final image size after img aug
    res_size = (128, 128)

    """ Initialization
        --------------
    """
    # Create the experiment folder structure
    exp_folder = create_exp_folder(output_path)
    make_folder(os.path.join(exp_folder, 'logs'))
    make_folder(os.path.join(exp_folder, 'model'))
    # Copy source file to experiment folder for reproducability
    shutil.copytree('src/', os.path.join(exp_folder, 'src_exp'))
    # Create the model
    model = inception.inception_v3()
    model.cuda()
    print(model)
    # Image augmentation functions
    train_aug_funcs = [
            rotate,
            crop_hand,
            flipLR,
            lambda img,kp_2D,prng: resize(img, kp_2D, res_size)
            ]
    test_aug_funcs = [
            crop_hand,
            lambda img, kp_2D, prng: resize(img, kp_2D, res_size)
            ]
    # Create the data queues
    data_dim = (128, 128, 3)
    train_queue = DataQueue(ds_train_path, queue_size, batch_size, n_workers, 
            data_dim, train_aug_funcs, valid_frac=0.1)
    train_queue.start_workers()
    test_queue = DataQueue(ds_test_path, queue_size, batch_size, x_dim=data_dim, 
            aug_funcs=test_aug_funcs, test=True)
    test_queue.start_workers()
    # The loss function
    loss = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam([
        {'params' : model.parameters()},
        ], lr=learning_rate
    )
    # The logger
    logger = LogInterface(os.path.join(exp_folder, 'logs'))
    # The trainer
    trainer = Trainer(model, optimizer, loss, train_queue, test_queue, logger, 
            exp_folder, checkpoint_path)
    # Start the training
    trainer.train(n_epochs, save_frequency, 100) 
    # Start the testing
    trainer.test()

