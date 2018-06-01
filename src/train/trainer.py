"""
    Contains the Trainer class that performs training and testing of a model
"""
import os
import shutil
import time
from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.util.misc import pyt2np, np2pyt
from src.util.visualization import draw_keypoints, concat_images


class Trainer():

    # TODO Make optional: logger
    def __init__(self, model, optimizer, loss, data_queue, test_queue, logger,
            exp_folder, checkpoint_path=None):
        """Initializes the trainer.

        Parameters
        ----------
        model: torch.nn.Module
            Model to train and test
        optimizer: torch.optim
            Optimizer to use
        loss: torch.nn.Loss
            Loss function of the model. Needs to return the average loss over 
            the batch, otherwise the loss returned will not be correct.
        data_queue: DataQueue
            DataQueue containing the training and potentially validation 
            samples
        test_queue: DataQueue
            DataQueue containing the test samples
        logger: Logger
            Object that enables logging to tensorboard
        exp_folder: str
            Path to folder where experiment should be stored
        checkpoint_path: str, optional
            If specified, loads the optimizer and model configuration from the 
            file specified in checkpoint_path
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.data_queue = data_queue
        self.test_queue = test_queue
        self.logger = logger
        self.exp_folder = exp_folder
        # Load model and optimizer state if specified
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward_pass(self, batch_samples):
        """Perform one forward pass with the batch_samples. If optimizer is given,
        performs a backward pass too.

        Parameters
        ----------
        batch_samples: tuple(np.array)
            Tuple of np arrays, where the first element contains the input
            and the second element contains the targets

        Returns
        -------
        loss_out: float
            Returns the average loss
        model_out: torch.tensor
            The output of the model wrt to the input
        """
        # Get the model input and the corresponding targets
        model_in = np2pyt(batch_samples[0]).cuda()
        targets = np2pyt(batch_samples[1]).cuda()
        # Perform a forward pass
        model_out = self.model(model_in)
        loss_out = self.loss(model_out, targets)

        return loss_out, model_out

    def backward_pass(self, loss_out):
        """Performs the backward pass of the model wrt loss_out

        Parameters
        ----------
        loss_out: torch.tensor
            torch tensor containing the output of the loss of the model which 
            one wants to backpropagate
        """
        # Compute the gradients
        loss_out.backward()
        # Update the model parameters
        self.optimizer.step()
        self.optimizer.zero_grad()


    def run_one_epoch(self, n_iter, data_queue, phase, log_interval=50, update_params=True):
        """Perform n_iter forward_pass and summarizes statistics over all
        the iterations. If update_params=true, then performs a backward_pass
        per forward_pass.

        Parameters
        ----------
        n_iter: int
            Number of iterations over the training data to perform
        data_queue: DataQueue
            DataQueue that contains the data to iterate over and pass to the
            model.
        phase: str
            Indicates in which phase (e.g 'train','valid') the epoch is run in. 
            Used for logging purposes.
        log_interval: int, optional
            Number of iterations per logging event.
        update_params: bool
            Indicates if the parameters are to be updated for each forward_pass
            via backward_pass

        Returns
        -------
        mean_loss: float
            The mean loss over all data samples.
        """
        total_loss = 0
        n_samples_processed = 0
        for c_iter in range(n_iter):
            t_iter = time.perf_counter()
            # Current batch
            batch_samples = data_queue.get()
            t_dequeue = time.perf_counter() - t_iter
            # Perform one forward pass
            loss_out, model_out = self.forward_pass(batch_samples)
            # Perform one backward pass
            if update_params:
                self.backward_pass(loss_out)
            loss_out = pyt2np(loss_out)
            c_batch_size = batch_samples[0].shape[0]
            # Undo the loss averaging. This is to ensure correct loss calculation
            # when the batch sizes are not all the same size
            loss_out *= c_batch_size
            # Handle the iteration summary
            total_loss += loss_out
            # Time taken for one training iteration
            t_iter = time.perf_counter() - t_iter
            # Number of samples processed since start of epoch
            n_samples_processed += c_batch_size
            if (c_iter % log_interval) == 0:
                c_mse_loss = loss_out / c_batch_size

                print(('It: %04d/%04d. MSE Loss: %3.5f. Dequeue time: %.5f ms. ' \
                    + 'It. time: %.5f ms') % \
                    (c_iter, n_iter, c_mse_loss, t_dequeue,t_iter)
                )

        # Get the loss averaged over all samples processed
        mean_loss = total_loss / n_samples_processed
        print(phase + ': MSE loss %f' % mean_loss)
        # Logging
        self.logger.log_scalar(phase + ': MSE loss', mean_loss)
        # Plot the images to tensorboard
        img_arr = []
        for idx in range(9):
            c_img = batch_samples[0][idx]
            c_kp_2D = pyt2np(model_out[idx])
            c_img_draw = draw_keypoints(c_img, c_kp_2D)
            img_arr.append(c_img_draw)
        img_concat = concat_images(img_arr, dim=np.array([3,3])) 
        self.logger.log_image(phase + ": batch sample" ,img_concat)

        return mean_loss
    
    def train(self, n_epochs, save_frequency, n_iter_train=None):
        """Performs the training of the model for n_epochs epochs. Each epoch
        has n_iter_train iterations over the training set. If data_queue
        contains a valid set, it is run after each epoch. The model and optimizer
        state are saved every save_frequency epochs or if a new best error is
        found.

        Parameters
        ----------
        n_epochs: int
            Number of epochs to train the model
        save_frequency: int,
            Number of epochs after which to save the model.
        n_iter_train: int, optional
            Number of iterations per epoch. If not specified will, set it to 
            n_samples / batch_size such that (almost) the entire dataset is 
            iterated over.
        """
        if not n_iter_train:
            n_iter_train = self.data_queue.n_train // self.data_queue.batch_size
        model_best_err = float('inf')
        for epoch in range(1, n_epochs + 1):
            print("##### Epoch %d/%d #####" % (epoch, n_epochs))
            print("Training:")
            # Perform training
            model_err = self.run_one_epoch(n_iter_train, 
                    self.data_queue.train_queue, 'train')
            if self.data_queue.has_valid:
                n_iter_valid = self.data_queue.n_valid // self.data_queue.batch_size
                print("Validating:")
                # Perform validation
                model_err = self.run_one_epoch(n_iter_valid, 
                        self.data_queue.valid_queue, 'valid', update_params=False)
            # Periodically save the model or if the best performance is achieved
            if (epoch % save_frequency) == 0 or model_err < model_best_err:
                if model_err < model_best_err:
                    is_best = True
                    model_best_err = model_err
                else:
                    is_best = False
                self.save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict_model' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'model_err' : model_err
                    }, is_best,
                    file_dir=os.path.join(self.exp_folder, 'model'),
                    file_name="checkpoint_epoch_%d.pth" % epoch)



    def test(self):
        """Runs the model on the all data contained in test_queue without updating
        the parameter values
        """
        n_iter_test = ceil(self.test_queue.n_samples / self.test_queue.batch_size)
        print("Testing:")
        print("Number of test samples: %d" % self.test_queue.n_samples)
        test_err = self.run_one_epoch(n_iter_test, self.test_queue, 'test', 
                update_params=False)

    def save_checkpoint(self, state, is_best, file_dir, file_name='checkpoint.pth'):
        """Save the current state of the model
        """
        save_path = os.path.join(file_dir, file_name)
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(file_dir, 'model_best.pth'))

    def load_checkpoint(self, file_dir):
        """Loads checkpoint file specified by path file_dir and loads the state
        of the model and optimizer into the current object.

        Parameters
        ----------
        file_dir: str
            Path to the checkpoint file containing the model and optimizer state

        """
        print(file_dir)
        if os.path.isfile(file_dir):
            print("Loading checkpoint: %s" % file_dir)
            checkpoint = torch.load(file_dir)
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded of epoch %d" % start_epoch)
        else:
            print("No checkpoint found. Nothing loaded")
