import numpy as np
from tensorflow.keras.utils import Sequence
from utils.helper import glob_all_files, random_patch, show_images, paths2numpy, images_cropper
from utils.helper import face_resize_augmentation, random_imaug


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

    def __init__(self, fg_folder, bg_folder, batch_size):
        """
        :param fg_folder: Tight 하게 Crop 된 월리 이미지들이 들어 있는 폴더
        :param bg_folder: 월리가 있는 부분은 block 처리된 full image 가 들어 있는 폴더
        """

        # 이미지가 존재하는 경로를 카운트하여 배치 사이즈를 구하는데 사용하기 위한 변수입니다.
        self.path_count = 0

        # 포그라운드 이미지들을 저장할 리스트형태의 인스턴스 변수 생성
        self.fg_imgs = []
        # 월리 얼굴을 경로로 가져온다.
        paths = glob_all_files(fg_folder)

        # 가져온 경로에서 이미지 파일을 numpy 배열로 바꾼다.
        for path in paths:
            images = paths2numpy(path)

            # 현재 경로의 폴더가 비어있을 경우 바로 다음 경로를 확인한다.
            if not images:
                continue

            # 비어있는 경로가 아닐 경우 경로 카운트를 1 추가
            self.path_count += 1

            images = face_resize_augmentation(images)  # numpy 배열의 월리 얼굴들을 사이즈별로 새로 생성한다.
            self.fg_imgs.append(images)

        # 경로 내에 이미지가 없을 경우 프로그램이 중단한다.
        assert self.path_count > 0, print("올바른 경로가 아니거나, 모든 경로 내에 얼굴 이미지가 존재하지 않습니다.")

        # 배치 사이즈를 입력받은 배치 사이즈 * 찾고자하는 캐릭터들의 수+1 로 설정한다.
        self.batch_size = batch_size * (self.path_count + 1)
        """
        현재 crop 사이즈를 외부에서 입력받은 값을 기준으로 하기 때문에 주석처리 합니다.
        
        # 월리 얼굴의 최대 높이, 최대 너비를 구한다.
        (max_h, min_h), (max_w, _), (_, _) = image_info(self.fg_imgs)
        # 높이와 너비 중 더 큰 것을 max_length 에 저장한다.
        max_length = np.maximum(max_h, max_w)
        # stride size 를 설정한다.
        stride = int(max_length/4)
        """

        # background 이미지의 경로를 가져온다.
        self.bg_imgs = glob_all_files(bg_folder)

        assert self.bg_imgs, print("올바른 경로가 아니거나, 경로 내에 배경 이미지가 존재하지 않습니다.")
        
        # 가져온 경로에서 이미지 파일을 numpy 로 바꾼다.
        self.bg_imgs = paths2numpy(self.bg_imgs)
        # 크기가 (36, 36) 인 filter 를 (10, 10) 단위로 background 이미지를 분할한다.
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
        # 현재 인스턴스에 저장된 배치사이즈는 실제 사용된 백그라운드 이미지 갯수 * (포그라운드 이미지 종류 수 + 1) 이 된 상태기 때문에
        # 해당 배치사이즈에 (포그라운드 이미지 종류 수 + 1) 만큼을 다시 나누어 정확한 배치사이즈를 구할 수 있도록 한다.
        return np.int(np.ceil(len(self.bg_imgs) / (self.batch_size / (self.path_count + 1))))

    def __getitem__(self, index):
        """
        인스턴스를 인덱싱 했을 때, 인덱스에 해당하는 이미지들과 라벨들을 리턴하는 매직 메소드.
        """
        # 분할된 background 이미지의 인덱싱 범위를 지정한다.
        start = index * int(self.batch_size / (self.path_count + 1))
        end = (index + 1) * int(self.batch_size / (self.path_count + 1))

        # background 인덱싱 후 길이에 맞게 라벨을 붙인다.
        # background label: 0(배경)
        bg_imgs = self.bg_imgs[start: end]
        bg_labs = np.zeros(len(bg_imgs))

        # foreground 이미지에 해당하는 배치 사이즈를 지정한다.
        fg_batch = len(bg_imgs)

        # foreground 이미지와 라벨값이 저장되는 리스트
        fg_imgs = []
        fg_labs = []

        for i in range(self.path_count):
            # 랜덤하게 월리 얼굴을 가져오기 위한 fg_batch 개수 만큼의 인덱스를 무작위로 가져온다.
            indices = np.random.choice(np.arange(len(self.fg_imgs[i])), size=fg_batch)
            # 위에서 지정한 랜덤 인덱스에 맞는 월리 얼굴을 가져온다.
            patch_imgs = [self.fg_imgs[i][ind] for ind in indices]
            # 길이에 맞게 라벨을 붙인다.
            # foreground label: 1(월리), 2(여자친구), 3(마법사), 4(가짜 월리)
            patch_labs = np.full(len(patch_imgs), i + 1)

            # foreground 이미지들을 여러가지 형태로 변환시킵니다.
            patch_imgs = random_imaug(patch_imgs)

            # 기존의 월리 얼굴을 background 의 랜덤한 위치에 붙여준 후, 그 이미지를 foreground 이미지로 다시 저장한다.
            for ind, img in enumerate(bg_imgs.copy()):
                patch_imgs[ind] = random_patch(img, patch_imgs[ind])

            fg_imgs.append(patch_imgs)
            fg_labs.append(patch_labs)

        # 포그라운드 이미지가 여러 종류인 경우, 아래에서 백그라운드와 concatenate 를 진행시켜주기 위해 미리 포그라운드들 끼리 합쳐준다.
        concat_fg_x = np.concatenate(fg_imgs)
        concat_fg_y = np.concatenate(fg_labs)

        # batch_xs 에는 background, foreground 의 이미지들이 들어간다.
        batch_xs = np.concatenate([concat_fg_x, bg_imgs], axis=0)
        # batch_ys 에는 background, foreground 의 라벨들이 들어간다.
        batch_ys = np.concatenate([concat_fg_y, bg_labs], axis=0)

        batch_ys = batch_ys.astype(np.int8)

        return batch_xs, batch_ys


if __name__ == '__main__':
    train_fg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/wally_face_tight_crop'
    train_bg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/block_imgs'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])
