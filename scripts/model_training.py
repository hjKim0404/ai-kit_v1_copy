# 1. 작업환경을 현재 위치로 옮긴다.
import os
BASE_PATH = os.path.dirname(os.getcwd())
os.chdir(BASE_PATH)

# 2. 필요한 패키지들을 불러온다.
import numpy as np
import matplotlib.pyplot as plt

from utils.helper import show_images, glob_all_files, paths2numpy, cropper, draw_rectangles
from utils.datagenerator import TightFaceProvider

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model, save_model, load_model

# 3. 학습용 데이터를 불러온 후 인공지능 모델 학습에 알맞게 가공해준다.
fg_folder = './data/wally_face_tight_crop'
bg_folder = './data/block_imgs'

tfp = TightFaceProvider(fg_folder, bg_folder, batch_size=64)


# 4. 학습 모델을 구축한다.
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
fcn = Dense(units=256, activation='relu')(flat)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

fcn = Dense(units=256, activation='relu')(relu)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

pred = Dense(units=1, activation='sigmoid')(relu)


# 모델 생성
model = Model(inputs, pred)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

""" 
**test: 모델 학습 및 저장 부분. 현재는 모델을 불러와서 작업을 하고 있기 때문에 임시 주석처리 합니다.
# 5. 모델을 학습시킨다.
for i int range(1):
    model.fit_generator(tfp, epochs=1)
    
# 6. 모델을 저장한다.
model.save("./models/best_model2")
"""

# 7. 이미 저장된 모델이 있을 경우, 그 모델을 불러온다.
model = load_model("./models/new_model")

# 8. 검증용 데이터를 불러온다.
val_folder = "./data/full_images_val"
images = glob_all_files(val_folder)
images = paths2numpy(images)

bucket_crop_imgs = []
bucket_crop_crds = []

for image in images:
    cropped_imgs, cropped_crds = cropper(image, 10, 10, 35, 35)
    bucket_crop_imgs.append(cropped_imgs)
    bucket_crop_crds.append(cropped_crds)


# 9. 모델을 검증용 데이터로 테스트합니다.
for i, image in enumerate(images):
    cropped_imgs = bucket_crop_imgs[i]
    cropped_crds = bucket_crop_crds[i]

    # 예측값을 저장한 후, 그 중 0.5가 넘는 값들에 대한 불리언 마스크를 만드는 부분입니다.
    predicts = model.predict(cropped_imgs)
    bool_mask = (predicts > 0.5)[:, 0]

    show_images(cropped_imgs[bool_mask])    # 불리언 마스크를 적용시킨 결과로 얻은 월리의 얼굴로 추정되는 이미지 조각들을 출력
    target_crds = np.array(cropped_crds)[bool_mask]     # 월리의 얼굴이 있을 것으로 예상되는 좌표들을 저장

    predicts = predicts[bool_mask]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장
    result_image = draw_rectangles(image, target_crds, (255, 0, 0), 3, predicts[:, 0])

    plt.imshow(result_image)
    plt.show()
