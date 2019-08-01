import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image
from keras.applications.vgg19 import preprocess_input


def load_image(path_to_image, reshaped_size=(400, 300)):
    image = imread(path_to_image)
    # Reshape image to 400 x 300
    reshaped_image = np.array(Image.fromarray(image).resize(size=reshaped_size))
    # Add 1 dimension to image array
    reshaped_image = np.expand_dims(reshaped_image, axis=0)
    # Preprocess input to match model's expected input
    preprocessed_image = preprocess_input(reshaped_image)
    return preprocessed_image


def add_noise_to_image(image, noise_ratio=0.9):
    """
    Add noise to image based on the noise ratio
    :param image: original image as numpy array
    :param noise_ratio: between 0 and 1.0. Default 0.9
    :return:
    """

    # Generate a random noise_image
    image_shape = image.shape
    noise = np.random.uniform(-20, 20, image_shape).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    output = noise * noise_ratio + image * (1 - noise_ratio)

    return output

