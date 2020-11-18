import numpy as np
from tensorflow.keras.utils import Sequence
from helper import images_cropper, paths2numpy
from helper import glob_all_files, random_patch, show_images
from augmentator import face_resize_augmentation
from analysis import image_info


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

        # Load foreground images
        self.fg_imgs = glob_all_files(fg_folder)
        self.fg_imgs = paths2numpy(self.fg_imgs)
        self.fg_imgs = face_resize_augmentation(self.fg_imgs)

        # Get size from foreground
        (max_h, min_h), (max_w, min_), (_, _) = image_info(self.fg_imgs)
        max_length = np.maximum(max_h, max_w)

        # Load background images
        self.bg_imgs = glob_all_files(bg_folder)
        self.bg_imgs = paths2numpy(self.bg_imgs)
        stride = int(max_length/4)
        self.bg_imgs = images_cropper(self.bg_imgs, stride, stride, max_length+1, max_length+1)

        # shuffle images
        np.random.shuffle(self.bg_imgs)

    def __len__(self):
        """
        Descirption:
        모든 background 이미지를 한번씩 불러오기 위한 index 길이를 반환하는 함수
        1,2,3,4,5,6 이미지, batch size 가 4 라면
        bg : 1,  2 | 3, 4 | 5, 6
        fg : 1', 2'| 3' 4' | 5' 6'
        return images : 1 1' 2 2' | 3 3' 4 4' | 5 5' 6 6'
        해당 형식처럼 되어야 하기 때문에 len(self.bg_imgs) / (self.batch_size/2) 가 되어야 합니다.

        :return: int, 1 epochs 당 step 수
        """
        return np.int(np.ceil(len(self.bg_imgs) / (self.batch_size/2)))

    def __getitem__(self, index):
        start = index * np.int(self.batch_size/2)
        end = (index + 1) * np.int(self.batch_size/2)

        # Background
        bg_imgs = self.bg_imgs[start: end]
        bg_labs = np.zeros(len(bg_imgs))

        # Foreground
        n_fg_batch = self.batch_size - len(bg_imgs)
        indices = np.random.choice(np.arange(len(self.fg_imgs)), size=n_fg_batch)
        fg_imgs = [self.fg_imgs[ind] for ind in indices]
        fg_labs = np.ones(len(fg_imgs))

        # background image 에 foreground 을 random 한 위치에 붙인다.
        # TODO : foreground 가 background 보다 같거나 커야 하는지 보증하는 코드가 있어야 한다.
        for ind, img in enumerate(bg_imgs.copy()):
            fg_imgs[ind] = random_patch(img, fg_imgs[ind])

        batch_xs = np.concatenate([fg_imgs, bg_imgs], axis=0)
        batch_ys = np.concatenate([fg_labs, bg_labs], axis=0)
        return batch_xs, batch_ys


if __name__ == '__main__':
    train_fg_folder = './images/wally_face_tight_crop'
    train_bg_folder = './images/block_imgs'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[3][0])
    print(tfp[3][0].shape)
    print(tfp[3][1].shape)
