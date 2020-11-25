import os
from glob import glob
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from utils.augmentator import apply_aug
from utils.helper import paths2pil, pil2numpy, resize_images, images_cropper, rect2square, video2image, paths2numpy
from utils.helper import glob_all_files, random_patch, show_images
from utils.augmentator import face_resize_augmentation
from utils.analysis import image_info


class CropperProvider(Sequence):
    def __init__(self, cropped_images, batch_size=1):
        self.cropped_images = cropped_images
        self.batch_size = batch_size

    def __len__(self):
        return np.int(len(self.cropped_images) / self.batch_size)

    def __getitem__(self, index):
        start = index * np.int(self.batch_size)
        end = (index + 1) * np.int(self.batch_size)
        return self.cropped_images[start : end],  np.zeros(len(self.cropped_images[start : end]))


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
        self.bg_imgs, _ = images_cropper(self.bg_imgs, stride, stride, max_length+1, max_length+1)

        # shuffle images
        np.random.shuffle(self.bg_imgs)

    def __len__(self):
        return np.int(np.ceil(len(self.bg_imgs) / self.batch_size))

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

        for ind, img in enumerate(bg_imgs.copy()):
            fg_imgs[ind] = random_patch(img, fg_imgs[ind])

        batch_xs = np.concatenate([fg_imgs, bg_imgs], axis=0)
        batch_ys = np.concatenate([fg_labs, bg_labs], axis=0)
        return batch_xs, batch_ys


class VideoTrainProvider(object):
    """
    위 클래스는 아래와 같은 수순으로 수행됩니다.
    # Foreground 생성
        1. 비디오에서 Foreground 이미지를 추출합니다.
        2. Foreground 이미지를 하나의 텐서로 생성합니다.
    # Background 생성
        1. 검정색으로 block 이 된 Full Image 이미지를 특정 크기로 잘라 하나의 numpy 로 생성합니다.
    """
    def __init__(self, fg_folder, bg_folder, resize):
        # load foreground video
        fg_paths = glob(os.path.join(fg_folder, '*'))

        # generate foreground
        self.fg_imgs = []
        for path in fg_paths[:]:
            imgs = video2image(path)
            imgs = resize_images(imgs, resize)
            imgs = rect2square(imgs)
            self.fg_imgs.append(imgs)
        self.fg_imgs = np.concatenate(self.fg_imgs, axis=0)

        # load background image
        bg_paths = glob(os.path.join(bg_folder, '*'))

        # generate foreground
        bg_imgs = paths2pil(bg_paths[:])
        bg_imgs = pil2numpy(bg_imgs)
        self.bg_imgs, _ = images_cropper(bg_imgs, 30, 30, 120, 120)


class VideoValidationProvider(object):
    """
    위 클래스는 Validation 데이터셋을 생성합니다.

    # Foreground 생성
        1. 이미 생상된 이미지를 가져와 Numpy 로 생성합니다.
    # Background 생성
        1. 검정색으로 block 이 된 Full Image 이미지를 특정 크기로 잘라 하나의 numpy 로 생성합니다.
    """

    def __init__(self, fg_folder, bg_folder, stride_size, crop_size):
        """

        :param fg_folder: 이미 만들어놓은 Validation Image 가 들어 있는 폴더
        :param bg_folder: 월리가 있는 부분은 block 처리된 full image 가 들어 있는 폴더
        :param stride_size: (H, W)
        :param crop_size: (H, W)
        """

        # load foreground
        self.fg_imgs = glob_all_files(fg_folder)
        self.fg_imgs = paths2numpy(self.fg_imgs, resize=crop_size)
        self.fg_imgs = np.array(self.fg_imgs)
        self.fg_labs = np.ones(len(self.fg_imgs))

        # generate background
        self.bg_imgs = glob_all_files(bg_folder)
        self.bg_imgs = paths2numpy(self.bg_imgs)
        self.bg_imgs = np.array(self.bg_imgs)
        self.bg_labs = np.zeros(len(self.bg_imgs))

        self.bg_imgs, _ = images_cropper(self.bg_imgs, stride_size[0], stride_size[1], crop_size[0], crop_size[1])


class WallyProvider(Sequence):
    def __init__(self, batch_size, root_foler='./images'):
        self.batch_size = batch_size
        # load fg images
        fg_folder = os.path.join(root_foler, 'foreground', '*.jpg')
        fg_paths = glob(fg_folder)

        # foreground resize #
        self.fg_imgs = np.array([np.array(Image.open(path).resize([32, 32])) for path in fg_paths])
        self.fg_labs = np.ones(len(self.fg_imgs))

        # load bg images
        bg_folder = os.path.join(root_foler, 'background', '*')
        bg_paths = glob(bg_folder)
        self.bg_imgs = np.concatenate([np.load(path) for path in bg_paths[:3]], axis=0)
        self.bg_labs = np.zeros(len(self.bg_imgs))

    def generate_background_batch(self):
        indices = np.arange(len(self.bg_labs))
        np.random.shuffle(indices)
        return self.bg_imgs[indices[:self.batch_size]], self.bg_labs[indices[:self.batch_size]]

    def __len__(self):
        return np.int(np.ceil(len(self.bg_labs) / self.batch_size))

    def __getitem__(self, index):
        self.batch_bg_imgs, self.batch_bg_labs = self.generate_background_batch()
        batch_imgs = np.concatenate([self.fg_imgs, self.batch_bg_imgs], axis=0)
        batch_labs = np.concatenate([self.fg_labs, self.batch_bg_labs], axis=0)

        return apply_aug(batch_imgs)/255., batch_labs


if __name__ == '__main__':

    # VideoValidationProvider Test
    # val_fg_folder = '/Users/seongjungkim/PycharmProjects/Wally/images/validation_foreground_images'
    # val_bg__folder = '/Users/seongjungkim/PycharmProjects/Wally/images/validation_background_images'
    # vvp = VideoValidationProvider(val_fg_folder, val_bg__folder, (32, 32), (240, 240))
    # print(vvp.fg_imgs.shape)
    # print(vvp.bg_imgs.shape)

    # VideoTrainProvider Test
    # train_fg_folder = '/Users/seongjungkim/PycharmProjects/Wally/video_sample'
    # train_bg_folder = '/Users/seongjungkim/PycharmProjects/Wally/images/block_imgs'
    # VideoTrainProvider(train_fg_folder, train_bg_folder, (240, 135))

    # TightFaceProvider Test
    train_fg_folder = './data/wally_face_tight_crop'
    train_bg_folder = './data/block_imgs_sample'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])

    pass
