import numpy as np
from tensorflow.keras.utils import Sequence
from utils.helper import glob_all_files, random_patch, show_images


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

    def __len__(self):
        return np.int(np.ceil(len(self.bg_imgs) / self.batch_size))

    def __getitem__(self, index):

        return batch_xs, batch_ys


if __name__ == '__main__':
    train_fg_folder = './data/wally_face_tight_crop'
    train_bg_folder = './data/block_imgs_sample'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])