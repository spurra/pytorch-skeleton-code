"""This script reads and visualizes samples from the dataset created
by create_cropped_data.py
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

target_dir = "./data/processed/evaluation"

anno_path = os.path.join(target_dir, 'anno_evaluation_cropped.pickle')
img_bin_path = os.path.join(target_dir, 'img_evaluation_cropped.bin')
img_size = (128, 128)
# The size in bytes of one image
img_size_bin = 3 * img_size[0] * img_size[1]

with open(anno_path, 'rb') as f:
    anno_eval = pickle.load(f)

with open(img_bin_path, 'rb') as f:
    n_test_samples = int(os.stat(img_bin_path).st_size / img_size_bin)

    print("Number of test samples %d" % n_test_samples)
    for i in range(0, n_test_samples):
        if (i % 100) == 0:
            print("Sample %d" % i)
            # Go to the correct file position
            f.seek(i*img_size_bin)
            img_bin = f.read(img_size_bin)
            # Transform binaries in to a numpy array. Reshape it into the
            # in the same dimension as it was saved it (CxHxW)
            img = np.frombuffer(img_bin, dtype='uint8').reshape(
                3, img_size[0], img_size[1]
            )
            # The image needs to be HxWxC for matplotlib to plot it
            img = img.transpose(1,2,0)
            kp_2D = anno_eval[i]
            plt.imshow(img)
            plt.plot(kp_2D[:,0], kp_2D[:,1], 'rx')
            plt.show()
            plt.pause(0.001)


