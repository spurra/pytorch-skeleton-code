"""
    Offers methods that return queues that can make use of multiprocessing
"""
import numpy as np
from numpy.random import RandomState
import os
import time

import h5py
import torch
import torch.multiprocessing as mp

''' TODO 
Implement a dataqueue manager class that returns a train and validation queue
Run augment_data function to see what final dim is instead of requesting as arg (both x_dim,y_dim)
Comment functions and classes
If the way validation queues are implemented wastes to much memory due to valid_queue size, implement
a new worker_func which only concentrates on the validation queue.
- Put both queues into a queue array. That way, theres no confusion with the name
'''

class DataQueue():

    def __init__(self, data_path, queue_size, batch_size, n_workers=1, x_dim=None, 
            aug_funcs=None, test=False, memory=False, valid_frac=0, prng_seed=1):
        """Initializes the DataQueue object by assigning all its parameters
        Parameters
        ----------
        data_path: str
            Path to file that contains the data. Needs to be a hdf5 file.
        queue_size: int
            Size of the queue which holds the data batches. If its too big,
            the system may run out of memory. If its too small, the trainer may
            have to wait until the next batch is processed.
        batch_size: int
            Size of one batch.
        n_workers: int
            Number of processes that work concurrently on loading samples.
        x_dim: tuple(int), optional
            Dim of the input samples. If not given, it will load it from
            the hdf5 file that contains the data. If aug_func is given, need to
            be careful that x_dim corresponds to the RESULTING input dim after
            applying the augmentation function.
        aug_func: list(function), optional
            Contains data augmentation functions which are applied to the samples
            in the order given by the list.
        test: bool, optional
            Indicates if data corresponds to the test set. If true, DataQueue 
            sets n_workers=1 and only performs one pass over the test set in 
            sequential order. This guarantees that always the same data are 
            processed in the same order.
        memory: bool, optional
            Indicates if the entire dataset should be preloaded into memory 
            instead of kept on disk.
        valid_frac: float, optional
            The fraction of data which is kept as validation set.
        prng_seed: int, optional
            Random number generator seed value used for any randomness that
            DataQueue has.

        """
        self.train_queue = mp.Queue(queue_size)
        self.worker_pool = []
        # The HDF5 file containing the dataset
        self.data_path = data_path
        if memory:
            hdf5 = h5py.File(data_path, 'r', driver='core')
        else:
            hdf5 = h5py.File(data_path, 'r')
        n_samples = hdf5.attrs['n_samples']
        if not x_dim:
            self.x_dim = hdf5.attrs['x_dim']
        else:
            self.x_dim = x_dim
        self.y_dim = hdf5.attrs['y_dim']
        self.hdf5 = hdf5
        # If test is enabled, traverse the dataset sequentially
        if test:
            n_workers = 1
            # Create the train idx in reverse so that we can use the pop method
            self.train_idx = np.arange(n_samples-1, -1, -1)
            self.has_valid = False
            n_valid = 0
            n_train = n_samples
        self.test = test
        if not test:
            if valid_frac > 0:
                # Create a validation queue 
                self.has_valid = True
                # Round down
                n_valid = int(valid_frac * n_samples - 0.5)
                n_train = n_samples - n_valid
                # Create the train and valid indeces
                idx = np.random.permutation(n_samples)
                # Make the validation queue big enough
                valid_queue_size = 2*n_valid // batch_size
                self.valid_queue = mp.Queue(valid_queue_size)
                self.train_idx = idx[:n_train]
                self.valid_idx = list(idx[n_train:])
            else:
                self.has_valid = False
                self.train_idx = np.random.permutation(n_samples)
                n_train = n_samples
                n_valid = 0
        self.train_idx = list(self.train_idx)
        self.n_valid = n_valid
        self.n_train = n_train
        self.n_samples = n_samples
        self.aug_funcs = aug_funcs
        self.n_workers = n_workers
        self.are_working = mp.Value('b', False)
        self.batch_size = batch_size
        self.prng = RandomState(prng_seed)

    def __len__(self):
        return self.train_queue.qsize()


    def get(self, ds_split=0):
        """Returns one batch from the data split specified in ds_split
        
        Parameters
        ----------
        ds_split: int, optional
            The split from which the data sample is returned

        Returns
        -------
        data_batch: np.array
            Batch of data samples from the specified split ds_split
        """
        if ds_split==0:
            data_batch = self.train_queue.get()
        else:
            data_batch = self.valid_queue.get()
        return data_batch


    def start_workers(self):
        """Creates workers as Processes, having the __worker_thread as target
        and starts them.
        """

        self.are_working.value = True
        # Single threaded execution for debugging purposes
        # self.__worker_proc(self.prng.randint(0,100))
        for i in range(self.n_workers):
            curr_worker = mp.Process(name='Worker %d' % i,
                target=self.__worker_proc,
                args=(self.prng.randint(0, 100),))
            # Mark it daemon such that it does not block the main thread from
            # exiting.
            curr_worker.daemon = True
            curr_worker.start()
            self.worker_pool.append(curr_worker)


    def stop_workers(self):
        """Terminates the worker processes started by start_workers.
        """
        self.are_working.value = False


    def __worker_proc(self, seed):
        """The worker process function runs on an infinite loop and inserts
        data batches into the data queue. To terminate the workers, call the 
        function stop_workers. It first fills in data batch of training data.
        If the training queue is full, it waits until it can insert its batch.
        Once successful, attempts to fill a data batch of validation
        samples. If the validation queue is full, it drops the batch and 
        continues filling the training queue.

        Parameters
        ----------
        seed: int
            The seed value of the RandomState that the worker_proc use
        """
        prng = RandomState(seed)
        self.prng = prng
        x_dim = self.x_dim
        y_dim = self.y_dim
        batch_size = self.batch_size

        while self.are_working.value:
            data_batch = [
                # Have it in CxHxW
                np.zeros(
                    (batch_size, x_dim[2], x_dim[0], x_dim[1])
                    ),
                np.zeros(
                    (batch_size, y_dim[0]//2, y_dim[1]-1)
                    ),
            ]
            # Sampled idx wrt the batch
            idx_batch = 0
            try:
                while idx_batch < batch_size:
                    idx = self.__get_idx()
                    data = self.__load_data(idx)
                    if self.aug_funcs:
                        data = self.__augment_data(data)
                    idx_batch = self.__add_to_batch(data, data_batch, idx_batch)
                # Push it in the data queue.
                # WARNING: This could cause deadlocks if valid_queue isnt big enough
                self.train_queue.put(data_batch)
            except IndexError:
                # For test data. If there are no more idx to load, terminate
                self.stop_workers()
            if self.has_valid and not self.valid_queue.full():
                # Fill the validation queue tmp
                data_batch = [
                    # Have it in CxHxW
                    np.zeros(
                        (batch_size, x_dim[2], x_dim[0], x_dim[1])
                        ),
                    np.zeros(
                        (batch_size, y_dim[0]//2, y_dim[1]-1)
                        ),
                ]
                # Sampled idx wrt the batch
                idx_batch = 0
                while idx_batch < batch_size:
                    idx = self.__get_idx(1)
                    data = self.__load_data(idx)
                    if self.aug_funcs:
                        data = self.__augment_data(data)
                    idx_batch = self.__add_to_batch(data, data_batch, idx_batch)
                try:
                    self.valid_queue.put_nowait(data_batch)
                except Queue.Full:
                    pass
        print("%s terminated" % mp.current_process().name)


    def __get_idx(self, ds_split=0):
        """Returns the idx of the next data sample to load wrt to the data split. 
        If DataQueue is not specified as test, it will return a random idx from 
        [0, n_samples]. If DataQueue is specified as the test set, it will 
        return the next sample in sequence.

        Parameters
        ----------
        ds_split: int, optional
            Split for which an idx is sampled

        Returns
        -------
        idx: int
            Index of the next data sample to load
        """
        # Select the current training set
        if ds_split==0:
            curr_idx = self.train_idx
            n_samples = self.n_train
        else:
            curr_idx = self.valid_idx
            n_samples = self.n_valid
        # Select idx
        if self.test:
            idx = curr_idx.pop()
        else:
            idx = curr_idx[self.prng.randint(0, n_samples)]

        return idx


    def __load_data(self, idx):
        """Returns data sample tuple indexed by idx, where its first element is 
        the input and the second is the target. The data point should be brought
        into a form ready to be processed by the augmentation functions.
        
        Parameters
        ----------
        idx: int
            The index of the sample to load

        Returns
        -------
        samples: tuple(np.array)
            Tuple where first element is the input and the second is the target.
        """
        # Load the data from the HDF5 file
        img = self.hdf5['img'][idx]
        kp_2D = self.hdf5['kp_2D'][idx]
        # Map the data from [0, 255] to [0,1]
        img = img / 255

        samples = []
        for c_kp_2D in [kp_2D[:21], kp_2D[21:]]:
            # Only include the visible hands
            if c_kp_2D[:,2].sum() == 21:
                samples.append((img, c_kp_2D[:,:2]))

        return samples

    def __augment_data(self, data):
        """Perform data augmentation contained in aug_funcs in the order specified
        by aug_func.

        Parameters
        ----------
        data: tuple(np.array)
            Tuple where first element is input and the second is the target
        """
        # Target resize size
        data_mod = []
        for data_pair in data:
            img = data_pair[0]
            kp_2D = data_pair[1]
            for aug_func in self.aug_funcs:
                img, kp_2D = aug_func(img, kp_2D, self.prng)
            data_mod.append((img, kp_2D))

        return data_mod

    def __add_to_batch(self, data, data_batch, idx_batch):
        """Push the data sample contained in data into the batch specified by
        data_batch at the index specified by idx_batch
        
        Parameters
        ----------
        data: tuple(np.array)
            Tuple where first element is input and the second is the target
        data_batch: np.array
            Container of the current batch
        idx_batch: index of where to insert data into data_batch

        Returns
        -------
        idx: int
            Index of the next position to insert future data
        """
        # Push it in the current data batch
        for data_pair in data:
            if idx_batch < data_batch[0].shape[0]:
                # Bring it in CxHxW so that its ready to be processed.
                data_batch[0][idx_batch] = data_pair[0].transpose(2,0,1)
                data_batch[1][idx_batch] = data_pair[1]
                idx_batch += 1

        return idx_batch


# Test DataQueue
if __name__ == '__main__':
    import matplotlib
    #matplotlib.use('Agg') # Comment this when not working remote
    import matplotlib.pyplot as plt
    import os
    import sys
    sys.path.append(os.getcwd())
    from src.util.img_transformations import (crop_hand, resize, check_coords,
        flipLR, rotate)

    res_size = (128, 128)
    x_dim = (128, 128, 3)
    queue_size = 1000
    n_threads = 4
    batch_size = 2
    ds_path = './data/processed/training/training_set.h5'
    train_aug_funcs = [
            rotate,
            crop_hand,
            flipLR,
            lambda img,kp_2D,prng: resize(img, kp_2D, res_size)
        ]
    dq = DataQueue(ds_path, queue_size, batch_size, n_threads, x_dim, 
        aug_funcs=train_aug_funcs, test=True)
    print(len(dq))
    dq.start_workers()
    print('Sleeping')
    time.sleep(5)
    dq.stop_workers()
    for i in range(queue_size + n_threads):
        data_sample = dq.get()
        for j in range(batch_size):
            c_img = data_sample[0][j]
            c_anno = data_sample[1][j]
            c_img = c_img.transpose(1,2,0)

            plt.imshow(c_img)
            plt.plot(c_anno[:,0], c_anno[:,1], 'rx')
            print("Queue sample %d. Batch sample %d" % (i,j))
            plt.show()
    for i in range(dq.n_valid):
        data_sample = dq.get('valid')
        for j in range(batch_size):
            c_img = data_sample[0][j]
            c_anno = data_sample[1][j]
            c_img = c_img.transpose(1,2,0)

            plt.imshow(c_img)
            plt.plot(c_anno[:,0], c_anno[:,1], 'rx')
            print("Queue sample %d. Batch sample %d" % (i,j))
            plt.show()
    # time.sleep(1)
    # print(dq.train_queue.full())
