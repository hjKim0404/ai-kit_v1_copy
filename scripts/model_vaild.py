# 1. 필요한 패키지들을 불러옵니다.
import os
BASE_PATH = os.path.dirname(os.getcwd())
os.chdir(BASE_PATH)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from utils.helper import cropper, glob_all_files, paths2numpy, draw_rectangles


# 2. 학습 시킨 모델을 불러옵니다
model = load_model('./models/best_model')
model.summary()


# 3. 평가용 데이터를 불러옵니다.
paths = glob_all_files('./data/full_images_val/')

imgs = paths2numpy(paths)

bucket_crop_imgs = []
bucket_crop_crds = []
for img in imgs:
    cropped_images, cropped_coords = cropper(img, 10, 10, 34, 34)
    bucket_crop_imgs.append(cropped_images)
    bucket_crop_crds.append(cropped_coords)


# 4. 모델을 평가용 데이터로 테스트합니다.
# 잘린 이미지 중 월리가 있는 이미지를 찾고 해당 이미지의 좌표를 원본 이미지에 출력합니다
for img_ind, im in enumerate(imgs):
    cropped_imgs = bucket_crop_imgs[img_ind]
    cropped_crds = bucket_crop_crds[img_ind]

    # Wally 라고 생각되는 이미지의 index을 가져옵니다.
    predicts = model.predict(cropped_imgs)
    bool_mask = (predicts > 0.5)[:, 0]

    # Wally 라고 생각되는 이미지 좌표를 가져옵니다.
    target_crds = np.array(cropped_crds)[bool_mask]

    # 전체 이미지에 월리라고 예측 되는 부분에 사각형을 그립니다.
    predicts = predicts[bool_mask]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장합니다.
    result_image = draw_rectangles(im, target_crds, (255, 0, 0), 3, predicts[:, 0])

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.show()