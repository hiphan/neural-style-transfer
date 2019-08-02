import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras import backend as K


# compute content cost of image C and S at the chosen layer
def content_cost(content_image, generation_image, model, layer='block3_pool'):
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

        aS = intermediate_model.predict(style_image)
        aG = intermediate_model.predict(generation_image)

        aS_tensor = K.variable(aS)
        aG_tensor = K.variable(aG)

        layer_J = style_cost(aS_tensor, aG_tensor)

        weighted_style_J = weighted_style_J + layer_J * weight

    return weighted_style_J


# Weighted total cost
def total_cost(model, weight_dict, content_image, style_image, generation_image, alpha=10, beta=40):

    # Wrap loss function expected by Keras
    def cost_fn(y_true, y_pred):
        content_J = content_cost(content_image, style_image, model)
        style_J = weighted_style_cost(model, style_image, generation_image, weight_dict)
        total_J = alpha * content_J + beta * style_J
        return total_J

    return cost_fn
