"""
    Provides functionality for visualizing data
"""
import matplotlib.pyplot as plt
import numpy as np

def draw_keypoints(img, kp_2D):
    """Plots the 2D keypoints using matplotlib and returns a numpy array
    containing the image
    """

    if img.shape[0]==1 or img.shape[0]==3:
        img = img.transpose(1,2,0)

    # Construct the plot
    fig = plt.figure()
    plt.imshow(img)
    plt.plot(kp_2D[:,0], kp_2D[:,1], 'rx')
    plt.axis('off')
    plt.tight_layout(pad=0)
    # Draw it internally
    fig.canvas.draw()
    # Save it as numpy array
    img_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return img_np

def concat_images(img_arr, dim):
    """Concatenates all the images in img_arr to one image in the layout specified
    by dim

    Parameters
    ----------
    img_arr: list(np.array)
        A list of np arrays containing images. If the images arent in HxWxC 
        format, theyll be transposed. They are expected to be all of the 
        same size.
    dim: np.array
        Specifying the final dimensionality of the resulting image as a
        multiple of the image sizes contained in img_arr.

    Returns
    -------
    concat_img: np.array
        Contains the image which is a concatenation of all images contained in
        img_arr.
    """
    # Check if dim is consistent with img_arr
    assert(dim.prod() == len(img_arr))

    h,w,c = img_arr[0].shape
    concat_img = np.zeros((h*dim[0], w*dim[1], c))

    idx = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            c_img = img_arr[idx]
            # Transpose the images if theyre not in the correct format
            if c_img.shape[0]==1 or c_img.shape[0]==3:
                c_img = c_img.transpose(1,2,0)
            concat_img[i*h:(i+1)*h,j*w:(j+1)*w,:] = c_img
            idx += 1

    return concat_img

if __name__ == '__main__':
    import scipy.misc
    img = np.zeros((128, 128, 3))
    kp_2D = np.array([[64,64], [32, 32]])

    img_draw = draw_keypoints(img, kp_2D)

    scipy.misc.imsave("test.png", img_draw)

    a = np.zeros((10,5,3))
    b = np.ones((10,5,3))
    img_arr = [a,b]
    dim = np.array([1,2])

    img_concat = concat_images(img_arr, dim)
    plt.imshow(img_concat.squeeze())
    plt.show()
