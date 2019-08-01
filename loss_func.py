import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras import backend as K


# compute content cost of image C and S at the chosen layer
def content_cost(content_activation, generation_activation):
    m, h, w, c = K.int_shape(generation_activation)

    reshaped_C = K.reshape(content_activation, [h * w, c])
    reshaped_G = K.reshape(generation_activation, [h * w, c])

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

        style_arr = image.img_to_array(style_image)
        style_arr = np.expand_dims(style_arr, axis=0)
        style_arr = preprocess_input(style_arr)

        gen_arr = image.img_to_array(generation_image)
        gen_arr = np.expand_dims(gen_arr, axis=0)
        gen_arr = preprocess_input(gen_arr)

        aS = intermediate_model.predict(style_arr)
        aG = intermediate_model.predict(gen_arr)

        aS_tensor = K.variable(aS)
        aG_tensor = K.variable(aG)

        layer_J = style_cost(aS_tensor, aG_tensor)

        weighted_style_J = weighted_style_J + layer_J * weight

    return weighted_style_J


def cost(model, weight_dict, content_image, style_image, generated_image, alpha=10, beta=40):
    c_J = content_cost(content_image, style_image)
    s_J = weighted_style_cost(model, style_image, generated_image, weight_dict)
    total_J = alpha * c_J + beta * s_J
    return total_J
