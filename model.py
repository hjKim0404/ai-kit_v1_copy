from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.models import Model
from datagenerator import WallyProvider
from helper import cropper
import numpy as np
from PIL import Image
import sys
import os


class WallyModel(object):
    def __init__(self):
        pass;

    @staticmethod
    def bn_layer(x, filters):
        layer = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        return layer

    @staticmethod
    def conv_layer(x, filters):
        layer = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal', activation='relu')(x)
        return layer

    @staticmethod
    def block(x, filters, n_conv, layer_fn):
        for _ in range(n_conv):
            x = layer_fn(x, filters)(x)
        return MaxPooling2D()(x)


if __name__ == '__main__':

    wp = WallyProvider(44, os.path.join(sys.argv[1], 'images'))
    inputs = Input(shape=(32, 32, 3), name='inputs')

    conv = Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
    norm = BatchNormalization()(conv)
    relu = ReLU()(norm)
    pool = MaxPooling2D()(relu)

    conv = Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal')(pool)
    norm = BatchNormalization()(conv)
    relu = ReLU()(norm)
    pool = MaxPooling2D()(relu)

    conv = Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal')(pool)
    norm = BatchNormalization()(conv)
    relu = ReLU()(norm)
    pool = MaxPooling2D()(relu)

    flat = Flatten()(pool)

    #
    fcn1 = Dense(units=256, kernel_initializer='he_normal')(flat)
    norm = BatchNormalization()(fcn1)
    relu = ReLU()(norm)

    #
    fcn2 = Dense(units=256, activation='relu', kernel_initializer='he_normal')(relu)
    norm = BatchNormalization()(fcn1)
    relu = ReLU()(norm)

    #
    pred = Dense(1, activation='sigmoid')(relu)

    # Model
    model = Model(inputs, pred)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    #
    for i in range(2):
        model.fit_generator(wp, epochs=1)
        for page_number in [2,5,9,10,11,12]:
            path = os.path.join(sys.argv[1], 'images/block_imgs/{}.jpg'.format(page_number))
            img = np.array(Image.open(path))
            imgs = cropper(img, 5, 5, 32, 32)
            labs = np.zeros(len(imgs))

            model.evaluate(imgs, labs)