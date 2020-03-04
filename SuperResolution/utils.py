from keras.preprocessing.image import (
    img_to_array,
    array_to_img,
)

def drop_resolution(x, scale=3.0):
    # resize to small and resize to original in order to drop resolution easily.
    small_size = (int(x.shape[0] / scale), int(x.shape[1] / scale))
    img = array_to_img(x)
    small_img = img.resize(small_size, 3)
    return img_to_array(small_img.resize(img.size, 3))