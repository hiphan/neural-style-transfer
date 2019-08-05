import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import sys
from loss_func import *
from utils import *


import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras import backend as K


# compute content cost of image C and S at the chosen layer
def content_cost(content_image, generation_image, model, layer='block3_conv1'):
    # Get intermediate layer output
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    content_activation = intermediate_model.predict(content_image)
    generation_activation = intermediate_model.predict(generation_image)

    # Get shape
    assert len(generation_activation.shape) == 4
    m, h, w, c = generation_activation.shape

    # Unroll
    reshaped_C = K.reshape(content_activation, [h * w, c])
    reshaped_G = K.reshape(generation_activation, [h * w, c])

    # Compute cost
    content_J = (1 / (4 * h * w * c)) * K.sum(K.square(reshaped_C - reshaped_G))

    return content_J


def style_cost(style_activation, generation_activation):
    m, h, w, c = K.int_shape(generation_activation)

    def gram_mat(tensor):
        return K.dot(tensor, K.transpose(tensor))

    reshaped_S = K.transpose(K.reshape(style_activation, [h * w, c]))
    reshaped_G = K.transpose(K.reshape(generation_activation, [h * w, c]))

    gram_S = gram_mat(reshaped_S)
    gram_G = gram_mat(reshaped_G)

    style_J = K.square(1 / (2 * h * w * c)) * K.sum(K.square(gram_S - gram_G))
    return style_J


def weighted_style_cost(model, style_image, generation_image, weight_dict):
    weighted_style_J = 0

    for layer, weight in weight_dict.items():
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)

        style_activation = intermediate_model.predict(style_image)
        generation_activation = intermediate_model.predict(generation_image)

        style_activation_tensor = K.variable(style_activation)
        generation_activation_tensor = K.variable(generation_activation)

        layer_J = style_cost(style_activation_tensor, generation_activation_tensor)

        weighted_style_J = weighted_style_J + layer_J * weight

    return weighted_style_J


# Weighted total cost
def total_cost(model, weight_dict, content_image, style_image, generation_image, alpha=10, beta=40):
    content_J = content_cost(content_image, generation_image, model)
    style_J = weighted_style_cost(model, style_image, generation_image, weight_dict)
    total_J = alpha * content_J + beta * style_J
    return total_J

# Load VGG-19 model
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(300, 400, 3))
# print(vgg19_model.summary())

# Load content and style image
content_image = load_image("nature.jpg")
style_image = load_image("the_scream.jpg")

# Add noise to content image. Also optimizing objective
init_generation_image = add_noise_to_image(content_image)

# Create placeholder for initial generation image
generation_image = K.placeholder(shape=(1, 300, 400, 3), name='generation_output')

# Define layers' weights
layer_weights = {
    'block2_conv1': 0.3,
    'block3_conv1': 0.4,
    'block4_conv1': 0.3,
}


# Compute loss
def loss(generation_img):
    return total_cost(vgg19_model, layer_weights, content_image, style_image, generation_img)


def loss_fn(generation_img):
    fn = K.function([])

# Optimizer
opt = tf.train.AdamOptimizer()

# Train
epochs = 2
for _ in range(epochs):
    loss, grads = fn(init_generation_image)
    print(loss)
    print(grads)
    init_generation_image, _, _ = fmin_l_bfgs_b(func=total_cost, x0=init_generation_image, fprime=grads, maxfun=10)
    plt.imshow(init_generation_image)
    plt.show()


# Save generation image
