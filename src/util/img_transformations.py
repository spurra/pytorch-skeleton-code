"""
    Provides various image transformation functions
"""

import numpy as np
import cv2 as cv
import scipy.misc
import math

def check_coords(img, kp_2D):
    """Checks if the supplied 2D keypoints are contained fully in the image

    Parameters
    ---------
    img: np.array
        The image containing the hand
    kp_2D: np.array
        The 2D keypoints of the hand
    """
    x_min, x_max = math.ceil(kp_2D[:,0].min()), math.floor(kp_2D[:,0].max())
    y_min, y_max = math.ceil(kp_2D[:,1].min()), math.floor(kp_2D[:,1].max())
    if (x_min >= 0 and x_max <= img.shape[1] and y_min >=0 and
        y_max <= img.shape[0]):
        return True
    else:
        return False

def crop_hand(img, kp_2D, prng=None):
    """Crops the hand based on the 2D keypoints.

    The crop is created by placing a square bounding box around the hand based
    on the 2D keypoints provided. The bounding box is not tight, instead
    theres a buffer of 8 pixels to the outer most keypoint and its guaranteed
    to be contained in the image.

    Parameters
    ----------
    img : np.array
        An numpy array containing the image from which the hand will be
        cropped. The dimensions of img should be HxWxC.
    kp_2D : np.array
        The 2D coordinates of the hand joints wrt. the image in img

    Returns
    -------
    img_crop : np.array
        The cropped image of a hand
    kp_2D_crop: np.array
        The shifted 2D joint coordinates wrt. img_crop.
    """

    # Find the outer most keypoints to define the tight bounding box
    x_min, x_max = math.ceil(kp_2D[:,0].min()), math.floor(kp_2D[:,0].max())
    y_min, y_max = math.ceil(kp_2D[:,1].min()), math.floor(kp_2D[:,1].max())
    # Assert that the joints are all located within the image
    assert(check_coords(img, kp_2D))

    if x_max == img.shape[1]:
        x_max = img.shape[1]-1
    if y_max == img.shape[0]:
        y_max = img.shape[0]-1
    # Joint coordinate maxes
    left_max, right_max = x_min, x_max
    up_max, bottom_max = y_min, y_max
    # Add a buffer around the extracted bounding box
    buff = 8
    if x_min - buff >= 0:
        x_min -= buff
    if x_max + buff < img.shape[1]:
        x_max += buff
    if y_min - buff >= 0:
        y_min -= buff
    if y_max + buff < img.shape[0]:
        y_max += buff
    # Expand rectangle to square by elongating the shorter side of the rectangle
    y_diff = y_max - y_min
    x_diff = x_max - x_min
    if y_diff > x_diff:
        len_to_board_x = img.shape[1] - x_max
        # Check if square would extend beyond image. Extend on other side instead
        if x_min + y_diff < img.shape[1]:
            x_max = x_min + y_diff
        else:
            # Can't extend only on one side, extend on both sides
            x_min = x_max - (y_diff - (len_to_board_x - 1))
            x_max += (len_to_board_x - 1)
    else:
        len_to_board_y = img.shape[0] - y_max
        if y_min + x_diff < img.shape[0]:
            y_max = y_min + x_diff
        else:
            # Can't extend only on one side, extend on both sides
            y_min = y_max - (x_diff - (len_to_board_y - 1))
            y_max += (len_to_board_y - 1)

    if not prng is None:
        # Move the box randomly around the hand, such that it still includes the
        # whole hand
        left_range = min(x_max - right_max, x_min) - 1
        right_range = min(left_max - x_min, img.shape[1] - x_max) - 1
        x_translate = 0 if -left_range >= right_range else prng.randint(-left_range, right_range)
        up_range = min(y_max - bottom_max, y_min) - 1
        down_range = min(up_max - y_min, img.shape[0] - y_max) - 1
        y_translate = 0 if -up_range >= down_range else prng.randint(-up_range, down_range)
        x_min += x_translate
        x_max += x_translate
        y_min += y_translate
        y_max += y_translate

    # Extract the crop
    img_crop = img[y_min:y_max, x_min:x_max, :]

    assert(x_min >= 0 and x_max < img.shape[1] and y_min >=0 and
        y_max < img.shape[0])
    assert(y_min<y_max and x_min<x_max)

    # Translate the coordinates accordingly
    kp_2D_crop = kp_2D - np.array([[x_min, y_min]])
    # Some coordinates are slighty outside the image border. Fix
    if kp_2D_crop[:,0].max() > img_crop.shape[1]:
        idx = kp_2D_crop[:,0].argmax()
        kp_2D_crop[idx, 0] = img_crop.shape[1]
    if kp_2D_crop[:,1].max() > img_crop.shape[0]:
        idx = kp_2D_crop[:,1].argmax()
        kp_2D_crop[idx, 1] = img_crop.shape[0]

    return img_crop, kp_2D_crop

def resize(img, kp_2D, res_size):
    """Resizes the image to res_size and adjusts the 2D keypoints accordingly

    Parameters
    ----------
    img: np.array
        The image to be resized. Dimensions should be HxWxC
    kp_2D: np.array
        The 2D keypoints of the hand
    res_size: tuple
        The new image size

    Returns
    -------
    img_res: np.array
        The resized image
    kp_2D_res: np.array
        The adjusted 2D keypoints
    """

    # img_res = scipy.misc.imresize(img, res_size)
    img_res = cv.resize(img, res_size)
    # Y x X
    assert(img.shape[0] == img.shape[1] and res_size[0] == res_size[1])
    skew = [res_size[0] / img.shape[0], res_size[1] / img.shape[1]]

    # Adjust the 2D keypoints
    kp_2D_res = kp_2D * np.array(skew).reshape((1,2))

    return img_res, kp_2D_res

def rotate(img, kp_2D, prng):
    """Rotate the image with a random angle in [-45,45] degrees. The resulting
    image is guaranteed to contain all the keypoints.

    WARNING: Assuming root-centered joint coords for 3D and original coords
    for 2D

    Parameters
    ----------
    img: np.array
        The input image in shape HxWxC
    kp_2D: np.array
        Corresponding 2D hand joint coordinates
    prng: np.random.RandomState
        Random number generator for the random angle generation.

    Returns
    -------
    img: np.array
        The rotated image
    kp_2D: np.array
        The rotated 2D keypoints
    """
    rand_ang = prng.randint(-45, 45)
    # Shift hand into image center
    shift_val = [img.shape[1]/2 - kp_2D[0,0], img.shape[0]/2 - kp_2D[0,1]]
    shift_mat = np.array([[1, 0, shift_val[0]],[0,1, shift_val[1]], [0, 0, 1]])
    # Rotate counter-clockwise
    affine_mat = cv.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2),
        rand_ang, 1)
    affine_mat = np.matmul(affine_mat, shift_mat)
    img_rot = cv.warpAffine(img, affine_mat, (img.shape[1], img.shape[0]))
    # Rotate kp_2D accordingly
    kp_2D_rot = np.matmul(
        np.concatenate((kp_2D , np.ones((21,1))), axis=1),
        affine_mat.transpose()
        )

    # Check if rotation resulted in fingers being outside the image
    if check_coords(img_rot, kp_2D_rot):
        return img_rot, kp_2D_rot
    else:
        return img, kp_2D

def flipLR(img, kp_2D, prng):
    """Flips an image along the y-axis. The resulting img is guaranteed to 
    contain all the keypoints.

    Parameters
    ----------
    img: np.array
        The input image in shape HxWxC
    kp_2D: np.array
        Corresponding 2D hand joint coordinates
    prng: np.random.RandomState
        Random number generator for the random angle generation.

    Returns
    -------
    img: np.array
        The flipped image
    kp_2D: np.array
        The flipped 2D keypoints
    """
    if prng.binomial(1, 0.5) == 1:
        img_flip = cv.flip(img, 1)

        kp_2D_flip = kp_2D.copy()
        kp_2D_flip[:, 0] = kp_2D_flip[:, 0] - 2*(kp_2D[:,0]-img.shape[1]/2)
        return img_flip, kp_2D_flip
    else:
        return img, kp_2D

# Test the functions
if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    # Path to the input images and corresponding 2D keypoints
    input_path_train = 'data/processed/training/img_training.bin'
    target_path_train = 'data/processed/training/anno_training.pickle'

    with open(target_path_train, 'rb') as fio:
        anno = pickle.load(fio)


    n_samples = len(anno)
    img_dim = (3, 320, 320)
    img_res = (128, 128)
    prng = np.random.RandomState(1)
    with open(input_path_train, 'rb') as f:
        for i in range(n_samples):
            img_bin = f.read(img_dim[0] * img_dim[1] * img_dim[2])
            img = np.frombuffer(img_bin, dtype='uint8').reshape(
                img_dim[0], img_dim[1], img_dim[2]
            )

            img = img / 255
            # Reshape it to HxWxC due to the proceeding image transformations
            img = img.transpose(1,2,0)
            # Get the 2D keypoint annotation. Last column is visibility, which
            # we do not need.
            kp_2D = anno[i][:, :2]
            if (i%1000) == 0:
                print("Image %d/%d" % (i, n_samples))
            for c_kp_2D in [kp_2D[:21], kp_2D[21:]]:
                c_img = img

                if not check_coords(c_img, c_kp_2D):
                    continue

                c_img, c_kp_2D = rotate(c_img, c_kp_2D, prng)
                c_img, c_kp_2D = crop_hand(c_img, c_kp_2D, prng)
                c_img, c_kp_2D = flipLR(c_img, c_kp_2D, prng)
                c_img, c_kp_2D = resize(c_img, c_kp_2D, img_res)

                if prng.binomial(1, 0.001) == 1:
                    plt.imshow(c_img)
                    plt.plot(c_kp_2D[:,0], c_kp_2D[:,1], 'rx')
                    plt.show()

                if not check_coords(c_img, c_kp_2D):
                    print("Warning: kp_2D out of image plane for image %d" % i)
                    print("x_max: %.5f. y_max: %.5f" % (c_kp_2D[:,0].max(), c_kp_2D[:,1].max()))
                    print("x_min: %.5f. y_min: %.5f" % (c_kp_2D[:,0].min(), c_kp_2D[:,1].min()))
                    plt.imshow(c_img)
                    plt.plot(c_kp_2D[:,0], c_kp_2D[:,1], 'rx')
                    plt.show()
