import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras import backend as K
import sys
from utils import *
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))


# Load content and style image as Keras tensors
content_image = load_image("nature.jpg")
style_image = load_image("the_scream.jpg")

# Convert to np arrays to tensors
content_tensor = K.variable(content_image)
style_tensor = K.variable(style_image)

# Create placeholder for initial generation image
generation_image = K.placeholder(shape=(1, 300, 400, 3), name='generation_image')

# Define layers' weights
layer_weights = {
    'block2_conv1': 0.3,
    'block3_conv1': 0.4,
    'block4_conv1': 0.3,
}

# Load models
inputs = K.concatenate([content_tensor, style_tensor, generation_image], axis=0)
vgg19_model = VGG19(input_tensor=inputs, weights='imagenet', include_top=False)

# Get activation for each image
content_activation = dict((layer.name, layer.output[0, :, :, :]) for layer in vgg19_model.layers)
style_activation = dict((layer.name, layer.output[1, :, :, :]) for layer in vgg19_model.layers)
generation_activation = dict((layer.name, layer.output[2, :, :, :]) for layer in vgg19_model.layers)


# Compute content loss at one chosen layer
def content_loss(content_activation_list, generation_activation_list, layer='block3_conv1'):
    # Get activations at the chosen layer
    content_layer_activation = content_activation_list[layer]
    generation_layer_activation = generation_activation_list[layer]

    # Get shape
    assert len(generation_layer_activation.shape) == 3
    h, w, c = K.int_shape(generation_layer_activation)

    # Unroll
    reshaped_C = K.reshape(content_layer_activation, [h * w, c])
    reshaped_G = K.reshape(generation_layer_activation, [h * w, c])

    # Compute loss
    cl = (1 / (4 * h * w * c)) * K.sum(K.square(reshaped_C - reshaped_G))

    return cl


# Compute style loss at one layer
def style_loss(style_act, generation_act):
    # Get shape
    assert(len(generation_act.shape)) == 3
    h, w, c = K.int_shape(generation_act)

    # Compute the gram matrix of a tensor
    def gram_mat(tensor):
        return K.dot(tensor, K.transpose(tensor))

    # Unroll
    reshaped_S = K.transpose(K.reshape(style_act, [h * w, c]))
    reshaped_G = K.transpose(K.reshape(generation_act, [h * w, c]))

    # Get gram matrix
    gram_S = gram_mat(reshaped_S)
    gram_G = gram_mat(reshaped_G)

    # Compute loss
    sl = K.square(1 / (2 * h * w * c)) * K.sum(K.square(gram_S - gram_G))

    return sl


# Compute weighted style loss at different layers
def weighted_style_loss(style_activation_list, generation_activation_list, weights):
    wsl = 0

    for layer, weight in weights.items():
        curr_style_activation = style_activation_list[layer]
        curr_generation_activation = generation_activation_list[layer]

        layer_loss = style_loss(curr_style_activation, curr_generation_activation)

        wsl = wsl + layer_loss * weight

    return wsl


# Compute weighted total loss
alpha = 10
beta = 40
cl = content_loss(content_activation, generation_activation)
sl = weighted_style_loss(style_activation, generation_activation, layer_weights)
total_loss = alpha * cl + beta * sl

# Compute gradients
grads = K.gradients(loss=total_loss, variables=generation_image)

# Create function
fn = K.function([generation_image], ([total_loss] + grads))


def get_loss(generation_img):
    """
    Function to compute loss wrt the current generated image
    """
    generation_img = K.reshape(generation_img, [1, 300, 400, 3])
    return fn([generation_img])[0].astype('float64')


def get_grads(generation_img):
    """
    Function to compute gradients wrt the current generated image
    """
    generation_img = K.reshape(generation_img, [1, 300, 400, 3])
    return fn([generation_img])[1].flatten().astype('float64')


# Add noise to content image. Also optimizing objective
init_generation_image = add_noise_to_image(content_image)

# Training
iterations = 500
for iter in iterations:
    init_generation_image, _, _ = fmin_l_bfgs_b(func=get_loss, x0=init_generation_image.flatten(), fprime=get_grads)
    if iter % 10 == 9:
        result = restore_image(init_generation_image)
        img_path = 'nst_results/iteration_' + str(iter + 1) + '.png'
        plt.imshow(result)
        plt.savefig(img_path)
        plt.close()
