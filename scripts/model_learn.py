import numpy as np
from utils.helper import show_images, glob_all_files, paths2numpy, cropper
from utils.datagenerator import TightFaceProvider

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model, save_model

fg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/wally_face_tight_crop'
bg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/block_imgs'

tfp = TightFaceProvider(fg_folder, bg_folder, batch_size=64)

#sample_imgs = tfp[0][0]
#sample_labs = tfp[0][1]

#show_images(sample_imgs, sample_labs.tolist())



#모델 만들기
inputs = Input(shape=(35, 35, 3), name='inputs')

conv = Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal')(inputs)
norm = BatchNormalization()(conv)
relu = ReLU()(norm)
pool = MaxPooling2D()(relu)

conv = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal')(pool)
norm = BatchNormalization()(conv)
relu = ReLU()(norm)
pool = MaxPooling2D()(relu)

conv = Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal')(pool)
norm = BatchNormalization()(conv)
relu = ReLU()(norm)
pool = MaxPooling2D()(relu)

flat = Flatten()(pool)

#fully connected layer
layer = Dense(units=256, activation='relu')(flat)
layer = Dense(units=256, activation='relu')(layer)

pred = Dense(units=1, activation='sigmoid')(layer)


# 모델 생성
model = Model(inputs, pred)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


# 모델 검증하기
val_imgs = glob_all_files('/Users/pai/Downloads/Find_Wally_Deploy/data/full_images_val')
val_imgs = paths2numpy(val_imgs)

crop_imgs = []
crop_crds = []
for val_img in val_imgs:
    cropped_img, cropped_crd = cropper(val_img, 10, 10, 35, 35)
    crop_imgs.append(cropped_img)
    crop_crds.append(cropped_crd)

valid_img = np.concatenate(crop_imgs, axis=0)
print(valid_img.shape)

#model.fit_generator(tfp, epochs=1)
#indices = (model.predict(valid_img) > 0.5)[:, 0]
#show_images(valid_img[indices])

#model.save('/Users/pai/Downloads/Find_Wally_Deploy/models/test_model')
