import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import sys
from loss_func import *
from utils import *


# Load VGG-19 model
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(300, 400, 3))
# print(vgg19_model.summary())

# Load content and style image
content_image = load_image("nature.jpg")
style_image = load_image("the_scream.jpg")

# Add noise to content image
noised_content_image = add_noise_to_image(content_image)

# Inputs
x_1 = K.placeholder(name='content', shape=(None, 300, 400, 3))
x_2 = K.placeholder(name='style', shape=(None, 300, 400, 3))

# Optimizing objective
generation_image = K.variable(noised_content_image)

# Define loss function
weight_dict = {
    'block2_pool': 0.3,
    'block3_pool': 0.4,
    'block4_pool': 0.3,
}
print(total_cost(vgg19_model, weight_dict, content_image, style_image, noised_content_image))
loss = total_cost(vgg19_model, weight_dict, x_1, x_2, generation_image)

# Optimizer
opt = Adam()
updates = opt.get_updates(generation_image, [], loss)
train = K.function([x_1, x_2], [loss], updates=updates)

# Train
epochs = 2
for _ in range(epochs):
    _, _ = train([content_image, style_image])
    print(K.eval(generation_image))

# Save generation image
