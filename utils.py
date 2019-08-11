import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image
from keras.applications.vgg19 import preprocess_input
from keras import backend as K


def load_image(path_to_image, reshaped_size=(400, 300)):
    image = imread(path_to_image)
    # Reshape image to 400 x 300
    reshaped_image = np.array(Image.fromarray(image).resize(size=reshaped_size))
    # Add 1 dimension to image array
    reshaped_image = np.expand_dims(reshaped_image, axis=0)
    # Preprocess input to match model's expected input
    preprocessed_image = preprocess_input(reshaped_image)
    return preprocessed_image


def add_noise_to_image(image, noise_ratio=0.6):
    """
    Add noise to image based on the noise ratio
    :param image: original image as numpy array
    :param noise_ratio: between 0 and 1.0. Default 0.6
    :return:
    """

    # Generate a random noise_image
    image_shape = image.shape
    noise = np.random.uniform(-20, 20, image_shape).astype('float64')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    output = noise * noise_ratio + image * (1 - noise_ratio)

    return output


def restore_image(image):
    """
    Reshape and reverse VGG19's preprocessing
    """
    # Reshape
    restored_image = image.reshape(300, 400, 3)

    # Add mean pixel values
    color_means = [103.939, 116.779, 123.68]
    for c in range(3):
        restored_image[..., c] += color_means[c]

    # BGR to RGB
    restored_image = restored_image[..., ::-1]

    # Clip value between 0 and 255
    restored_image = np.clip(restored_image, a_min=0, a_max=255)

    # Convert pixel values to integer
    restored_image = restored_image.astype('uint8')

    return restored_image
