# 1. 작업환경을 현재 위치로 옮긴다.
import os
os.chdir(os.path.dirname(os.getcwd()))

# 2. 필요한 패키지들을 불러온다.
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model, load_model

from utils.datagenerator import TightFaceProvider
from utils.helper import glob_all_files, paths2numpy, images_cropper, draw_rectangles
from utils.helper import show_images


# 3. 학습용 데이터를 불러온 후 인공지능 모델 학습에 알맞게 가공해준다.
fg_folder = './data/train_imgs/waly_face'
bg_folder = './data/train_imgs/back_imgs'

tfp = TightFaceProvider(fg_folder, bg_folder, batch_size=64)


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
fcn = Dense(units=256, kernel_initializer='he_normal')(flat)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

fcn = Dense(units=256, kernel_initializer='he_normal')(relu)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

pred = Dense(units=1, activation='sigmoid')(relu)


# 모델 생성
model = Model(inputs, pred)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


# 5. 모델을 학습시킨다.

""" 
**test: 모델 학습 및 저장 부분. 현재는 모델을 불러와서 작업을 하고 있기 때문에 임시 주석처리 합니다.
model.fit(tfp, epochs=1)
    
# 6. 모델을 저장한다.
model.save("./models/waly_model2")

"""

# 7. 이미 저장된 모델이 있을 경우, 그 모델을 불러온다.

model = load_model("./models/waly_model")

# 8. 검증용 데이터를 불러온다.
val_folder = "./data/test_imgs"
paths = glob_all_files(val_folder)

assert paths, print("올바른 경로가 아니거나, 경로 내에 검증용 이미지가 존재하지 않습니다.")

imgs = paths2numpy(paths)

bucket_crop_imgs, bucket_crop_crds = images_cropper(imgs, 10, 10, 36, 36)

# 9. 모델을 검증용 데이터로 테스트합니다.
for i, img in enumerate(imgs):
    cropped_imgs = bucket_crop_imgs[i]
    cropped_crds = bucket_crop_crds[i]

    # 예측값을 저장한 후, 그 중 0.5가 넘는 값들에 대한 불리언 마스크를 만드는 부분입니다.
    predicts = model.predict(cropped_imgs)
    bool_mask = (predicts > 0.5)[:, 0]

    show_images(cropped_imgs[bool_mask])    # 불리언 마스크를 적용시킨 결과로 얻은 월리의 얼굴로 추정되는 이미지 조각들을 출력
    target_crds = np.array(cropped_crds)[bool_mask]     # 월리의 얼굴이 있을 것으로 예상되는 좌표들을 저장

    predicts = predicts[bool_mask]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장
    result_image = draw_rectangles(img, target_crds, (255, 0, 0), 3, predicts[:, 0])

    plt.imshow(result_image)
    plt.show()
