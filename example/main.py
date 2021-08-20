import tensorflow as tf
from tensorflow.keras.utils import plot_model


# RTX 3070 Set Session. ----------------------------------------
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)
# --------------------------------------------------------------

from models.fcn import fcn

if __name__ == '__main__':
    # defining the tensorflow distribute strategy
    # calling the model inside the scope

        model = fcn(image_size=(224, 224), ch_out=21, strides=32)
        # compiling the model with the optimizer, loss function and acc metrics
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()
        plot_model(model, '../fcn32s.png', show_shapes=True, show_layer_names=True)
