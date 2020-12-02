from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import glob_all_files, paths2numpy, cropper, draw_rectangles

model = load_model('/Users/pai/Downloads/Find_Wally_Deploy/models/best_model4')

# 모델 검증하기
paths = glob_all_files('/Users/pai/Downloads/Find_Wally_Deploy/data/full_images_val')
imgs = paths2numpy(paths)

bucket_crop_imgs = []
bucket_crop_crds = []
for img in imgs:
    crop_imgs, crop_crds = cropper(img, 10, 10, 35, 35)
    bucket_crop_imgs.append(crop_imgs)
    bucket_crop_crds.append(crop_crds)


for ind, img in enumerate(imgs):
    cropped_imgs = bucket_crop_imgs[ind]
    cropped_crds = bucket_crop_crds[ind]
    cropped_crds = np.array(cropped_crds)

    indices = (model.predict(cropped_imgs) > 0.5)[:, 0]

    for ind, val in enumerate(indices):
        if val == True:
            target_crds = cropped_crds[ind]
            img = draw_rectangles(img, target_crds, (255, 0, 0), 3)

            plt.imshow(img)
            plt.show()
