"""
    Script to turn all the image files of RHD into one binary dataset and
    only save the 2D keypoint annotation. This speeds up training.
"""
# Core lib
import pickle
import os
import numpy as np
import glob
# Additional libs
import matplotlib
matplotlib.use('Agg') # Comment out if not running remotely
import matplotlib.pyplot as plt
import cv2
import h5py

################ Main

plt.ion()
ds_set = 'test'
source_dir = './data/raw/%s' % ds_set
target_dir = './data/processed/%s' % ds_set
n_joints = 21
img_dim = (320, 320, 3)
kp_dim = (2*n_joints, 3)

# Load the annotations
with open(os.path.join(source_dir, 'anno_%s.pickle' % ds_set), 'rb') as fio:
    anno_all = pickle.load(fio)
# Only keep the 2D keypoints
anno = {}
for k,v in anno_all.items():
    anno[k] = v['uv_vis']
# Count the number of valid samples of hand images.
n_samples = len(anno)
# Create a HDF5 file
target_file = os.path.join(target_dir, '%s_set.h5' % ds_set)
if os.path.isfile(target_file):
    print('%s exists. Exiting.' % target_file)
    raise SystemExit
f_h5py = h5py.File(target_file, 'w')
f_h5py.attrs['n_samples'] = n_samples
f_h5py.attrs['x_dim'] = img_dim
f_h5py.attrs['y_dim'] = kp_dim
img = f_h5py.create_dataset('img', 
    (n_samples, img_dim[0], img_dim[1], img_dim[2]), dtype='B')
kp_2D = f_h5py.create_dataset('kp_2D', (n_samples, kp_dim[0], kp_dim[1]))
# Go through the images and store the imgs and its corresponding labels
idx = 0
for sample_name in sorted(glob.glob(source_dir + '/color/*.png')):
    # Extract the ID from the name
    sample_id = int(sample_name[-9:-4])
    # Load current image as color img
    c_img = cv2.imread(sample_name, 1)
    # Store it and its labels in the hdf5 file
    img[idx] = c_img
    kp_2D[idx] = anno[idx]
    if (sample_id % 5000) == 0:
        # Visualize some samples to see that they are consistent
        plt.imshow(img[idx])
        plt.plot(kp_2D[idx][:,0], kp_2D[idx][:,1], 'rx')
        plt.show()
        plt.pause(0.01)
        plt.clf()
        print("Visualizing sample %d" % sample_id)
    idx += 1

f_h5py.close()


