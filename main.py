from tensorflow.keras.models import Model, load_model
from utils.datagenerator import CropperProvider
from utils.helper import glob_all_files, paths2numpy, cropper, show_images
import numpy as np

# validation images 을 생성합니다.
# 정답 좌표는 아직 생성하지 않았습니다.
paths = glob_all_files('./data/full_images_val/')

# Load models
model = load_model('./models/')

# validation images 을 생성합니다.
# 정답 좌표는 아직 생성하지 않았습니다.
paths = glob_all_files('./data/full_images_val/')

crop_imgs = []
imgs = paths2numpy(paths)
for img in imgs:
    crop_imgs.append(cropper(img, 10, 10, 34, 34))
val_imgs = np.concatenate(crop_imgs, axis=0)

cropgen = CropperProvider(val_imgs, 6)

indices = (model.predict(cropgen) > 0.5)[:, 0]
show_images(val_imgs[indices])
