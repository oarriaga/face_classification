import numpy as np
import matplotlib.pyplot as plt


def make_mosaic(images, num_rows, num_cols, border=1, class_names=None):
    num_images = len(images)
    image_shape = images.shape[1:]
    mosaic = np.ma.masked_all(
        (num_rows * image_shape[0] + (num_rows - 1) * border,
         num_cols * image_shape[1] + (num_cols - 1) * border),
        dtype=np.float32)
    paddedh = image_shape[0] + border
    paddedw = image_shape[1] + border
    for image_arg in range(num_images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        image = np.squeeze(images[image_arg])
        image_shape = image.shape
        mosaic[row * paddedh:row * paddedh + image_shape[0],
               col * paddedw:col * paddedw + image_shape[1]] = image
    return mosaic


def normal_imshow(axis, data, vmin=None, vmax=None, cmap='gray',
                  axis_off=True):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    image = axis.imshow(
        data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    if axis_off:
        plt.axis('off')
    return image
