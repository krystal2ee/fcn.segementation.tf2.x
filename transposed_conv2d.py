from tensorflow.keras.layers import Conv2DTranspose
import tensorflow as tf

# RTX 3070 Set Session. ----------------------------------------
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)
# --------------------------------------------------------------

x = tf.ones((1,2,2,1))
print(x.shape)

y=Conv2DTranspose(filters=1, kernel_size=(3,3),
                  strides=(2,2))(x)
print(y)

print(y.shape)

y = Conv2DTranspose(filters=1,
                    kernel_size=(3,3),
                    strides=(1,1))(x)

print(y.shape)
