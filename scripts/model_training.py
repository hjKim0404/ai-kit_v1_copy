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
fg_folder = ['./data/book1/Wally/face_tight_crop', './data/book1/girlfriend/face_tight_crop',
             './data/book1/magician/face_tight_crop', './data/book1/fake/face_tight_crop']
bg_folder = './data/book1/common/block_imgs'

tfp = TightFaceProvider(fg_folder, bg_folder, batch_size=8)

# 4. 학습 모델을 구축한다.
inputs = Input(shape=(36, 36, 3), name='inputs')

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

pred = Dense(units=5, activation='softmax')(relu)

# 모델 생성
model = Model(inputs, pred)
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. 모델을 학습시킨다.
""" 
**test: 모델 학습 및 저장 부분. 현재는 모델을 불러와서 작업을 하고 있기 때문에 임시 주석처리 합니다.
model.fit(tfp, epochs=4)

# 6. 모델을 저장한다.
model.save("./models/multi_model")

"""

# 7. 이미 저장된 모델이 있을 경우, 그 모델을 불러온다.
model = load_model("./models/multi_model")


# 8. 검증용 데이터를 불러온다.
val_folder = "./data/book1/common/full_image_val"
images = glob_all_files(val_folder)
images = paths2numpy(images)

bucket_crop_imgs = []
bucket_crop_crds = []

for image in images:
    cropped_imgs, cropped_crds = cropper(image, 10, 10, 36, 36)
    bucket_crop_imgs.append(cropped_imgs)
    bucket_crop_crds.append(cropped_crds)

# 9. 모델을 검증용 데이터로 테스트합니다.
for i, image in enumerate(images):
    cropped_imgs = bucket_crop_imgs[i]
    cropped_crds = bucket_crop_crds[i]

    # 예측값을 저장한 후, 그 중 0.5가 넘는 값들에 대한 불리언 마스크를 만드는 부분입니다.
    predicts = model.predict(cropped_imgs)

    # 불리언 마스크를 만들어 적용시키기 위한 부분
    bool_mask = (np.argsort(predicts)[:, -1] != 0)

    # show_images(cropped_imgs[bool_mask])    # 불리언 마스크를 적용시킨 결과로 얻은 월리의 얼굴로 추정되는 이미지 조각들을 출력
    target_crds = np.array(cropped_crds)[bool_mask]     # 월리의 얼굴이 있을 것으로 예상되는 좌표들을 저장

    predicts = predicts[bool_mask]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장

    # # **test, 특정 오브젝트의 max 예측값의 인덱스를 받아와 출력하기 위한 부분
    # waly_max = np.where(predicts[:, 1] == np.max(predicts[:, 1]))
    # girl_max = np.where(predicts[:, 2] == np.max(predicts[:, 2]))
    # magi_max = np.where(predicts[:, 3] == np.max(predicts[:, 3]))
    # fake_max = np.where(predicts[:, 4] == np.max(predicts[:, 4]))
    #
    # target_crds = np.array(cropped_crds)[fake_max[0]]  # 월리의 얼굴이 있을 것으로 예상되는 좌표들을 저장
    #
    # predicts = predicts[fake_max[0]]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장

    predicts = np.max(predicts, axis=1)

    result_image = draw_rectangles(image, target_crds, (255, 0, 0), 3, predicts)

    plt.imshow(result_image)
    plt.show()
