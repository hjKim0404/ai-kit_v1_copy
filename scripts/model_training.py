# 1. 작업환경을 현재 위치로 옮긴다.
import os
os.chdir(os.path.dirname(os.getcwd()))

# 2. 필요한 패키지들을 불러온다.
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model

from utils.datagenerator import TightFaceProvider
from utils.helper import show_images, glob_all_files, paths2numpy, cropper, draw_rectangles

# 3. 학습용 데이터를 불러온 후 인공지능 모델 학습에 알맞게 가공해준다.
# 학습 및 검증에 사용될 월리 책 번호
book_number = 1

# 학습에 사용될 이미지들의 경로를 불러온다
fg_folder = [f'./data/book{str(book_number)}/Wally/face_tight_crop',
             f'./data/book{str(book_number)}/girlfriend/face_tight_crop',
             f'./data/book{str(book_number)}/magician/face_tight_crop',
             f'./data/book{str(book_number)}/fake/face_tight_crop']

bg_folder = f'./data/book{str(book_number)}/common/block_imgs'

tfp = TightFaceProvider(fg_folder, bg_folder, batch_size=8)

# # tfp가 정상적으로 생성되었는지 확인하기 위한 부분
# len(tfp)
#
# sample_imgs = tfp[0][0]
# sample_labs = tfp[0][1]
#
# show_images(sample_imgs, titles=sample_labs.tolist())

# 4. 학습 모델을 구축한다.
inputs = Input(shape=(36, 36, 3), name='inputs')

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

#fully connected layer
fcn = Dense(units=256, activation='relu')(flat)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

fcn = Dense(units=256, activation='relu')(relu)
norm = BatchNormalization()(fcn)
relu = ReLU()(norm)

# 5가지의 카테고리를 softmax를 사용하여 분류
pred = Dense(units=5, activation='softmax')(relu)

# 모델 생성
model = Model(inputs, pred)
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# 5. 모델을 학습시킨다.
""" 
**test: 모델 학습 및 저장 부분. 현재는 모델을 불러와서 작업을 하고 있기 때문에 임시 주석처리 합니다.
# 학습과 함께 사용될 검증 데이터를 불러오는 부분
test_paths = [f"./data/book{str(book_number)}/common/block_imgs_val",
              f"./data/book{str(book_number)}/wally/face_val",
              f"./data/book{str(book_number)}/girlfriend/face_val",
              f"./data/book{str(book_number)}/magician/face_val",
              f"./data/book{str(book_number)}/fake/face_val"]

imgs = glob_all_files(test_paths)

# 검증용 이미지들을 가져온다
test_images = []
for img in imgs:
    test_images.append(paths2numpy(img))

# 검증용 라벨들을 만든다
test_labels = []
for i in range(len(test_paths)):
    test_labels.extend(np.full(len(test_images[i]), i))

# list 형태의 test_images unlist 해주는 부분
test_images = [y for x in test_images for y in x]

test_images = np.array(test_images)
test_labels = np.array(test_labels)


hist = model.fit(tfp, epochs=7, validation_data=(test_images, test_labels))

# 6. 모델을 저장한다.

model.save("./models/no_model")

"""

# 7. 이미 저장된 모델이 있을 경우, 그 모델을 불러온다.
model = load_model("./models/book1_model")


# 8. 검증용 데이터를 불러온다.

val_folder = f"./data/book{str(book_number)}/common/full_image_val"
imgs = glob_all_files(val_folder)
imgs = paths2numpy(imgs)

bucket_crop_imgs = []
bucket_crop_crds = []

for img in imgs:
    cropped_imgs, cropped_crds = cropper(img, 10, 10, 36, 36)
    bucket_crop_imgs.append(cropped_imgs)
    bucket_crop_crds.append(cropped_crds)

# 9. 모델을 검증용 데이터로 테스트합니다.
for i, img in enumerate(imgs):
    cropped_imgs = bucket_crop_imgs[i]
    cropped_crds = bucket_crop_crds[i]

    # 예측값을 저장한 후, 그 중 0.5가 넘는 값들에 대한 불리언 마스크를 만드는 부분입니다.
    predicts = model.predict(cropped_imgs)

    # 불리언 마스크를 만들어 적용시키기 위한 부분
    # np.argsort(predicts)[:, -1] 이후,
    """
    != 0: 배경을 제외한 나머지 캐릭터들로 예측되는 곳은 true
    == 1: 월리로 예측되는 곳만 true
    == 2: 여자친구로 예측되는 곳만 true
    == 3: 마법사로 예측되는 곳만 true
    == 4: 가짜로 예측되는 곳만 true
    """
    bool_mask = (np.argsort(predicts)[:, -1] != 0) # & (np.max(predicts, axis=1) > 0.9)

    # show_images(cropped_imgs[bool_mask])    # 불리언 마스크를 적용시킨 결과로 얻은 월리의 얼굴로 추정되는 이미지 조각들을 출력

    target_crds = np.array(cropped_crds)[bool_mask]     # 찾고자 하는 캐릭터의 얼굴이 있을 것으로 예상되는 좌표들을 저장

    predicts = predicts[bool_mask]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장

    # # **test, 특정 오브젝트의 max 예측값의 인덱스를 받아와 출력하기 위한 부분
    # waly_max = np.where(predicts[:, 1] == np.max(predicts[:, 1]))
    # girl_max = np.where(predicts[:, 2] == np.max(predicts[:, 2]))
    # magi_max = np.where(predicts[:, 3] == np.max(predicts[:, 3]))
    # fake_max = np.where(predicts[:, 4] == np.max(predicts[:, 4]))
    #
    # # 모든 오브젝트의 max 예측값의 인덱스를 가져오는 부분
    # all_max = np.argsort(predicts*-1, axis=0)[0, 1:5]
    #
    # # 찾고자 하는 캐릭터의 얼굴이 있을 것으로 예상되는 좌표들을 저장
    # # [] 안에 넣는 값에 따라 찾고자 하는 캐릭터가 바뀌게 됩니다.
    # """
    # waly_max = 월리
    # girl_max = 월리 여자친구
    # magi_max = 마법사
    # fake_max = 가짜 월리
    # all_max = 위의 모든 캐릭터들
    # """
    # target_crds = np.array(cropped_crds)[all_max]
    #
    # predicts = predicts[all_max]  # 불리언 마스크를 적용시켰을 때의 예측값을 저장

    # 각 행들의 최대 예측값을 저장
    predicts = np.max(predicts, axis=1)

    result_image = draw_rectangles(img, target_crds, (255, 0, 0), 3, predicts)

    plt.figure(figsize=(20, 20))
    plt.imshow(result_image)
    plt.show()
