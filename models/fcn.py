import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *






def conv_relu(nout=4096, ks=3, stride=(1,1), pad='valid'):
    conv = Conv2D(filters=nout, kernel_size=ks, stride=stride,
        num_output=nout, padding=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    # return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
    return MaxPooling2D(bottom, pool_size=(ks, ks), strides=stride)



def fcn_32s(image_size, ch_in=3, n_classes=21):
    inputs = Input(shape=(*image_size, ch_in), name='input')

    # Building a pre-trained VGG-16 feature extractor (i.e., without the final FC layers)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    # Recovering the feature maps generated by each of the 3 final blocks:
    pool3 = vgg16.get_layer('block3_pool').output
    pool4 = vgg16.get_layer('block4_pool').output
    pool5 = vgg16.get_layer('block5_pool').output   # Feature Extractor's Output.



    # # the base net --VGG16
    # data, label = encoder(input_height=224,
    #                       input_width=224,
    #                       channels=3)
    # conv1_1, relu1_1 = conv_relu(data, 64, pad=100)
    # conv1_2, relu1_2 = conv_relu(relu1_1, 64)
    # pool1 = max_pool(relu1_2)
    #
    # conv2_1, relu2_1 = conv_relu(pool1, 128)
    # conv2_2, relu2_2 = conv_relu(relu2_1, 128)
    # pool2 = max_pool(relu2_2)
    #
    # conv3_1, relu3_1 = conv_relu(pool2, 256)
    # conv3_2, relu3_2 = conv_relu(relu3_1, 256)
    # conv3_3, relu3_3 = conv_relu(relu3_2, 256)
    # pool3 = max_pool(relu3_3)
    #
    # conv4_1, relu4_1 = conv_relu(pool3, 512)
    # conv4_2, relu4_2 = conv_relu(relu4_1, 512)
    # conv4_3, relu4_3 = conv_relu(relu4_2, 512)
    # pool4 = max_pool(relu4_3)
    #
    # conv5_1, relu5_1 = conv_relu(pool4, 512)
    # conv5_2, relu5_2 = conv_relu(relu5_1, 512)
    # conv5_3, relu5_3 = conv_relu(relu5_2, 512)
    # pool5 = max_pool(relu5_3)

    # fully conv -------------------------------------------------------------------------------------------------------
    # Replacing VGG dense layers by convolutions:

    x = Conv2D(filters=4096, kernel_size=7, padding='same')(pool5)     # Dense-1
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=4096, kernel_size=1, padding='same')(x)         # Dense-2
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    score_fr = Conv2D(filters=n_classes, kernel_size=1, padding='same')(x)       # Dense-3

    upscore = Conv2DTranspose(filters=n_classes, kernel_size=64, strides=32, padding='same', use_bias=False)(score_fr)

    outputs = Softmax()(upscore)

    fcn_model = Model(inputs, outputs)
    return fcn_model


def fcn(image_size, ch_in=3, ch_out=3, strides=8):

    inputs = Input(shape=(*image_size, ch_in), name='input')

    # Building a pre-trained VGG-16 feature extractor (i.e., without the final FC layers)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    # Recovering the feature maps generated by each of the 3 final blocks:
    f3 = vgg16.get_layer('block3_pool').output
    f4 = vgg16.get_layer('block4_pool').output
    f5 = vgg16.get_layer('block5_pool').output


    if strides == 32:
        f5_conv1 = Conv2D(filters=4096, kernel_size=7, padding='same',
                          activation='relu')(f5)
        f5_drop1 = Dropout(0.5)(f5_conv1)

        f5_conv2 = Conv2D(filters=4096, kernel_size=1, padding='same',
                          activation='relu')(f5_drop1)  # Dense-2
        f5_drop2 = Dropout(0.5)(f5_conv2)

        score_fr = Conv2D(filters=ch_out, kernel_size=1, padding='same')(f5_drop2)  # Dense-3

        upscore = Conv2DTranspose(filters=ch_out, kernel_size=64, strides=32, padding='same', use_bias=False)(
            score_fr)

        outputs = Softmax()(upscore)

        fcn_model = Model(inputs, outputs)
        return fcn_model

    if strides == 8:
        # Replacing VGG dense layers by convolutions:
        f5_conv1 = Conv2D(filters=4086, kernel_size=7, padding='same',
                          activation='relu')(f5)
        f5_drop1 = Dropout(0.5)(f5_conv1)
        f5_conv2 = Conv2D(filters=4086, kernel_size=1, padding='same',
                          activation='relu')(f5_drop1)
        f5_drop2 = Dropout(0.5)(f5_conv2)
        f5_conv3 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                          activation=None)(f5_drop2)

        # Using a transposed conv (w/ s=2) to upscale `f5` into a 14 x 14 map
        # so it can be merged with features from `f4_conv1` obtained from `f4`:
        f5_conv3_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                                      use_bias=False, padding='same', activation='relu')(f5)
        f4_conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                          activation=None)(f4)

        # Merging the 2 feature maps (addition):
        merge1 = add([f4_conv1, f5_conv3_x2])

        # We repeat the operation to merge `merge1` and `f3` into a 28 x 28 map:
        merge1_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                                    use_bias=False, padding='same', activation='relu')(merge1)
        f3_conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                          activation=None)(f3)
        merge2 = add([f3_conv1, merge1_x2])

        # Finally, we use another transposed conv to decode and up-scale the feature map
        # to the original shape, i.e., using a stride 8 to go from 28 x 28 to 224 x 224 here:
        outputs = Conv2DTranspose(filters=ch_out, kernel_size=16, strides=8,
                                  padding='same', activation=None)(merge2)


    fcn_model = Model(inputs, outputs)
    return fcn_model