import numpy as np
from tensorflow.keras.utils import Sequence
from utils.helper import glob_all_files, random_patch, show_images, paths2numpy, images_cropper
from utils.augmentator import face_resize_augmentation
from utils.analysis import image_info


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
        self.batch_size = batch_size

        # foreground 이미지를 resize 합니다.
        self.fg_imgs = glob_all_files(fg_folder)
        self.fg_imgs = paths2numpy(self.fg_imgs)
        self.fg_imgs = face_resize_augmentation(self.fg_imgs)

        # filter size, stride를 구합니다.
        (max_h, min_h), (max_w, _), (_, _) = image_info(self.fg_imgs)
        max_length = np.maximum(max_h, max_w)
        stride = int(max_length/4)

        # background 이미지를 가져와서 일정 간격으로 crop합니다.
        self.bg_imgs = glob_all_files(bg_folder)
        self.bg_imgs = paths2numpy(self.bg_imgs)
        self.bg_imgs, _ = images_cropper(self.bg_imgs, stride, stride, max_length+1, max_length+1)

        # background 이미지를 무작위로 섞습니다.
        np.random.shuffle(self.bg_imgs)


    def __len__(self):
        """
        한 epoch 당 스탭 수를 구해주는 매직 메소드입니다.
        """
        return np.int(np.ceil(len(self.bg_imgs)/ (self.batch_size/2)))


    def __getitem__(self, index):
        start = index * np.int(self.batch_size/2)
        end = (index+1) * np.int(self.batch_size/2)

        bg_imgs = self.bg_imgs[start: end]
        bg_labs = np.zeros(len(bg_imgs))

        n_fg_batch = self.batch_size - len(bg_imgs)
        indices = np.random.choice(np.arange(len(self.fg_imgs)), size=n_fg_batch)
        fg_imgs = [self.fg_imgs[ind] for ind in indices]
        fg_labs = np.ones(len(fg_imgs))

        for ind, img in enumerate(bg_imgs.copy()):
            fg_imgs[ind] = random_patch(img, fg_imgs[ind])

        # batch_size만큼 fg, bg 이미지를 가져옵니다.
        batch_xs = np.concatenate([fg_imgs, bg_imgs], axis=0)
        batch_ys = np.concatenate([fg_labs, bg_labs], axis=0)
        return batch_xs, batch_ys


if __name__ == '__main__':
    train_fg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/wally_face_tight_crop'
    train_bg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/block_imgs'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])