'''
    This script crops the images contained in evaluation set of RHD such that
    only the hand is shown.
'''

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math

from src.utility.img_transformations import crop_hand, resize, check_coords


plt.ion()

source_dir = "./data/raw/evaluation/"
target_dir = "./data/processed/evaluation"
crop_size = (128, 128)
n_kp = 21
idx_kp_c = 0

# Load the annotation file
with open(os.path.join(source_dir,'annotations.pickle'), 'rb') as f:
    anno_eval = pickle.load(f)

img_name_out = os.path.join(target_dir, 'img_evaluation_cropped.bin')
anno_name_out = os.path.join(target_dir, 'anno_evaluation_cropped.pickle')
# iterate samples of the set and write to binary file
with open(img_name_out, 'wb') as fo:
    n_samples = len(anno_eval.items())
    # There can be potentially up to 2*n_samples
    anno_eval_cropped = np.zeros((2*n_samples, n_kp, 2))
    for sample_id, anno in anno_eval.items():
        # Load current image
        img = scipy.misc.imread(os.path.join(source_dir, 'color', '%.5d.png' % sample_id))
        # Get the corresponding 2D keypoints
        kp_2D = anno['uv_vis'][:, :2]
        # Process left hand and then right hand
        for kp_2D_side in [kp_2D[:n_kp, :], kp_2D[n_kp:, :]]:
            img_side = img
            # Check if the current hand side is contained in the image
            if not check_coords(img_side, kp_2D_side):
                continue
            # Crop the image
            img_side, kp_2D_side = crop_hand(img_side, kp_2D_side)
            # Resize it to a uniform size
            img_side, kp_2D_side = resize(img_side, kp_2D_side, crop_size)
            # Write the image to file
            fo.write(
                img_side.transpose(2, 0, 1).reshape(1, img_side.shape[0],
                img_side.shape[1], img_side.shape[2])
            )
            # Add the cropped 2D keypoints
            anno_eval_cropped[idx_kp_c] = kp_2D_side
            idx_kp_c += 1

            if (sample_id % 100) == 0:
                # Plot the image to validate the cropping is correct.
                plt.imshow(img_side)
                plt.plot(kp_2D_side[:,0], kp_2D_side[:,1], 'rx')
                plt.show()
                plt.pause(0.01)
                plt.clf()
                print('%d / %d images done: %.3f percent' % \
                    (sample_id, n_samples, sample_id*100.0/n_samples)
                )


# Crop all the zero entries
anno_eval_cropped = anno_eval_cropped[:idx_kp_c]
with open(anno_name_out, 'wb') as f:
    pickle.dump(anno_eval_cropped, f)

print("Number of test samples: %d" % (idx_kp_c - 1))


