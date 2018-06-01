"""
    Script to evaluate on the test set
"""

import numpy as np
import os
from queue import Queue
import time
import shutil
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.append('./') # Because of the virtualenv in the servers
# Import this first, otherwise it will crash. (Loads first TF then torch)
from src.utility.log_interface import LogInterface
# import src.models.resnet as resnet
# import src.models.inception as inception
from src.training.data_handler import DataQueue
from src.utility.misc import pyt2np, np2pyt, create_exp_folder, make_folder
from src.utility.visualization import draw_keypoints

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd


def batch_iteration(batch_samples, model, loss, optimizer=None):
    """Perform one forward pass with the current batch. If optimizer is given,
    performs a backward pass too.

    """
    # Get the model input and the corresponding targets
    model_in = Variable(np2pyt(batch_samples[0]).cuda())
    targets = Variable(np2pyt(batch_samples[1]).cuda())
    # Perform a forward pass
    model_out = model(model_in)
    loss_out = loss(model_out, targets)
    if optimizer:
        # Compute the gradients
        loss_out.backward()
        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

    return loss_out, model_out

def iteration_loop(model, loss, data_queue, n_iter, logger, optimizer=None,
    log_interval=50):
    """Perform n_iter batch iterations and summarizes statistics over all
    the iterations.

    """

    # If an optimizer is given, the model is trained with it.
    if optimizer:
        model.train()
        stat_name = 'Train'
    else:
        model.eval()
        stat_name = 'Eval'

    model_out = []
    total_loss = 0
    for c_iter in range(n_iter):
        t_iter = time.perf_counter()
        # Current batch
        batch_samples = data_queue.get()
        t_dequeue = time.perf_counter() - t_iter
        # Perform one training iteration
        iter_summary = batch_iteration(batch_samples, model, loss, optimizer)
        # Handle the iteration summary
        total_loss += pyt2np(iter_summary[0])
        # Time taken for one training iteration
        t_iter = time.perf_counter() - t_iter
        if (c_iter % log_interval) == 0:
            c_mse_loss = pyt2np(iter_summary[0])
            print(('It: %03d/%03d. MSE Loss: %.5f. Dequeue time: %.5f ms. ' \
                + 'It. time: %.5f ms') % \
                (c_iter, n_iter, c_mse_loss, t_dequeue,t_iter)
            )

        model_out += list(pyt2np(iter_summary[1].data).reshape(-1, 2*21))


    # Logging
    avg_total_loss = total_loss / n_iter
    logger.log_scalar(stat_name + ': MSE loss', avg_total_loss)

    return avg_total_loss, model_out


if __name__ == '__main__':

    """ Options
        -------
    """
    # Path to the test images and corresponding 2D keypoints
    input_path_test = 'data/processed/evaluation/img_evaluation_cropped.bin'
    target_path_test = 'data/processed/evaluation/anno_evaluation_cropped.pickle'
    # Path to the output folder
    output_path = 'out'
    queue_size = 64
    data_dim = (3, 128, 128)
    n_workers = 1
    learning_rate = 1e-4
    batch_size = 1

    """ Initialization
        --------------
    """
    # Create the experiment folder structure
    exp_to_eval = 1
    exp_folder = 'out/exp_%03d' % exp_to_eval
    sys.path.append(exp_folder)
    # Import the model declaration from the experiment folder
    import src_exp.models.resnet as resnet
    # import src_exp.models.inception as inception
    checkpoint_epoch = 100
    checkpoint_epoch_file = 'checkpoint_epoch_%d.pth' % checkpoint_epoch
    # Easy baseline
    #model = inception.inception_v3()
    # Hard baseline
    model = resnet.resnet34()
    # Load the trained model
    state_dict = torch.load(os.path.join(exp_folder, 'model',
        checkpoint_epoch_file))
    model.load_state_dict(state_dict['state_dict_model'])
    model.cuda()
    # Create the data queues
    test_queue = DataQueue(input_path_test, target_path_test, queue_size,
        data_dim, batch_size, n_workers, data_aug=False, det=True)
    test_queue.start_workers()
    # The loss function
    loss = nn.MSELoss()
    # The logger
    logger = LogInterface(os.path.join(exp_folder, 'logs'))
    # Number of iterations per epoch
    n_iter_test = test_queue.n_samples // batch_size

    """ Testing
        --------
    """
    # The testing loop
    print("Testing:")
    # Perform testing
    avg_total_loss, model_out = iteration_loop(model, loss, test_queue, n_iter_test, logger)
    print(avg_total_loss)

    output_file = '%s/to_submit_to_kaggle_%d.csv' % (exp_folder, time.time())
    coloumns = []
    for i in range(21):
        coloumns += ['Joint %d x' % i]
        coloumns += ['Joint %d y' % i]

    final_output = pd.DataFrame(model_out, columns=coloumns)
    final_output.index.name = 'Id'
    final_output.to_csv(output_file)


