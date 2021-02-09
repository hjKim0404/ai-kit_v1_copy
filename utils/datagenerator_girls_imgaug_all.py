import numpy as np
import copy
from tensorflow.keras.utils import Sequence
from utils.helper import glob_all_files, paths2numpy, face_resize_augmentation, images_cropper
from utils.helper import random_patch
from utils.helper import show_images
from utils.helper import image_info
from utils.helper_girls_imgaug_all import random_imaug

# from utils.helper import image_info   # 현재 사용하지 않는 외부 함수이기 때문에 주석처리 합니다.


class TightFaceProvider(Sequence):
    """
    해당 클래스는 월리 얼굴을 Tight 하게 잘라 만든 데이터를 활용하기 위해 만들어진 DataGenerator 입니다.
    # Foreground 생성
        1. tight 이미지를 이미지내 모든 h, w에 대해 resize 합니다.
    # Background 생성

    Train Batch
        background 와 foreground 가 반반 들어갑니다.
    """
    def __init__(self, wally_folder, girl_folder, bg_folder, batch_size):
        """
        :param fg_folder: Tight 하게 Crop 된 월리 이미지들이 들어 있는 폴더
        :param bg_folder: 월리가 있는 부분은 block 처리된 full image 가 들어 있는 폴더
        """
        # 지정된 배치 사이즈
        self.batch_size = batch_size

        # 월리 얼굴을 경로로 가져온다.
        self.wally_imgs = glob_all_files(wally_folder)
        self.girl_imgs = glob_all_files(girl_folder)
        assert self.wally_imgs, print("올바른 경로가 아니거나, 경로 내에 얼굴 이미지가 존재하지 않습니다.")

        # 가져온 경로에서 이미지 파일을 numpy 배열로 바꾼다.
        self.wally_imgs = paths2numpy(self.wally_imgs)
        self.girl_imgs = paths2numpy(self.girl_imgs)
        # numpy 배열의 월리 얼굴들을 사이즈별로 새로 생성한다.
        self.wally_imgs = face_resize_augmentation(self.wally_imgs)
        self.aug_wally_imgs = random_imaug(self.wally_imgs.copy(), 0.8, 1.5, 1) 
        self.girl_imgs = face_resize_augmentation(self.girl_imgs)
        
        
        
        # **test: 현재 crop 사이즈를 외부에서 입력받은 값을 기준으로 하기 때문에 주석처리 합니다.
        
        # 월리 얼굴의 최대 높이, 최대 너비를 구한다.
        (max_h, min_h), (max_w, _), (_, _) = image_info(self.wally_imgs)
        # 높이와 너비 중 더 큰 것을 max_length에 저장한다.
        max_length = np.maximum(max_h, max_w)
        # stride size를 설정한다.
        stride = int(max_length/4)
        

        # background 이미지의 경로를 가져온다.
        self.bg_imgs = glob_all_files(bg_folder)

        assert self.bg_imgs, print("올바른 경로가 아니거나, 경로 내에 배경 이미지가 존재하지 않습니다.")

        # 가져온 경로에서 이미지 파일을 numpy로 바꾼다.
        self.bg_imgs = paths2numpy(self.bg_imgs)
        # 크기가 max_length인 filter를 stride 단위로 background 이미지를 분할한다.
        self.bg_imgs, _ = images_cropper(self.bg_imgs, 10, 10, 36, 36)

        # images_cropper로 인해 5차원 배열이 된 self.bg_imgs를 4차원 배열로 만들어주기 위한 부분
        self.bg_imgs = np.concatenate(self.bg_imgs, axis=0)

        # background 이미지를 무작위로 섞습니다.
        np.random.shuffle(self.bg_imgs)

    def __len__(self):
        """
        한 epoch 당 step 수를 구해주기 위한 매직 메소드.
        한 epoch 당 step 수: 전체 데이터 수  / 배치 크기
        """
        # 배치 사이즈가 절반인 이유는 step 수를 맞추기 위함
        return np.int(np.ceil(len(self.bg_imgs)/(self.batch_size/2)))

    def __getitem__(self, index):
        """
        인스턴스를 인덱싱 했을 때, 인덱스에 해당하는 이미지들과 라벨들을 리턴하는 매직 메소드.
        """
        # 분할된 background 이미지의 인덱싱 범위를 지정한다.
        start = index * int(self.batch_size/2)
        end = (index+1) * int(self.batch_size/2)

        # background 인덱싱 후 길이에 맞게 라벨을 붙인다.
        # background label: 0
        bg_imgs = self.bg_imgs[start: end].copy()
        bg_labs = np.zeros(len(bg_imgs))
        
        # girl_foreground 이미지를 생성한다.
        # index에 맞는 background 이미지를 copy
        girl_bg_imgs = self.bg_imgs[start: end].copy()
        # index 길이만큼 랜덤한 girl 이미지를 추출
        girl_size = len(girl_bg_imgs)
        girl_indices = np.random.choice(np.arange(len(self.girl_imgs)), girl_size, replace=False)
        girl_imgs = [self.girl_imgs[ind] for ind in girl_indices]
        # girl_imgs에 밝기 조절 전처리 사용
        girl_imgs = random_imaug(girl_imgs, 0.5 , 1.5, 0.5)
        # 추출한 girl 이미지와 background 이미지를 합친다.
        for ind, bg in enumerate(girl_bg_imgs):
            girl_bg_imgs[ind] = random_patch(bg, girl_imgs[ind])
        # girl_foreground 이미지의 label은 0으로 지정한다.
        girl_bg_labs = np.zeros(len(girl_bg_imgs))
        

        # foreground 이미지에 해당하는 배치 사이즈를 지정한다.
        fg_batch = len(bg_imgs)
        # 랜덤하게 월리 얼굴을 가져오기 위한 fg_batch 개수 만큼의 인덱스를 무작위로 가져온다.
        indices = np.random.choice(np.arange(len(self.wally_imgs)), size=fg_batch, replace=False)
        # 위에서 지정한 랜덤 인덱스에 맞는 월리 얼굴을 가져온다.
        fg_imgs = [self.wally_imgs[ind] for ind in indices]
        # 길이에 맞게 라벨을 붙인다.
        # foreground label: 1
        fg_labs = np.ones(len(fg_imgs))
        # 기존의 월리 얼굴을 background의 랜덤한 위치에 붙여준 후, 그 이미지를 foreground 이미지로 다시 저장한다.
        for ind, img in enumerate(bg_imgs.copy()):
            fg_imgs[ind] = random_patch(img, fg_imgs[ind])
            
        
        # 랜덤하게 월리 얼굴을 가져오기 위한 fg_batch 개수 만큼의 인덱스를 무작위로 가져온다.
        # 위에서 지정한 랜덤 인덱스에 맞는 월리 얼굴을 가져온다.
        aug_fg_imgs = [self.aug_wally_imgs[ind] for ind in indices]
        # 길이에 맞게 라벨을 붙인다.
        # foreground label: 1
        aug_fg_labs = np.ones(len(aug_fg_imgs))
        # 기존의 월리 얼굴을 background의 랜덤한 위치에 붙여준 후, 그 이미지를 foreground 이미지로 다시 저장한다.
        for ind, img in enumerate(bg_imgs.copy()):
            aug_fg_imgs[ind] = random_patch(img, aug_fg_imgs[ind])
        

        # batch_xs에는 background, foreground의 이미지들이 들어간다.
        batch_xs = np.concatenate([fg_imgs, aug_fg_imgs, girl_bg_imgs, bg_imgs], axis=0)
        # batch_ys에는 background, foreground의 라벨들이 들어간다.
        batch_ys = np.concatenate([fg_labs, aug_fg_labs, girl_bg_labs, bg_labs], axis=0)
        return batch_xs, batch_ys


if __name__ == '__main__':
    train_fg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/wally_face_tight_crop'
    train_bg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/block_imgs'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])
