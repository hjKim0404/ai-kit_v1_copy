import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

def show_count(i):
    raise NotImplemented

def save_images(save_folder, imgs):
    raise NotImplemented

def show_images(images, figsize=(10, 10), titles=None):
    raise NotImplemented

def get_name(path):
    raise NotImplemented

def get_names(paths):
    raise NotImplemented

def video2image(video_path, mode='rgb'):
    raise NotImplemented

def zero_padding(image, max_h, max_w):
    raise NotImplemented

def rect2square(np_images):
    raise NotImplemented

def cropper(img, stride_h=35, stride_w=35, filter_h=35, filter_w=35):
    raise NotImplemented
    return np.array(crop_imgs), coordinates


def images_cropper(images, stride_h=35, stride_w=35, filter_h=35, filter_w=35):
    raise NotImplemented
    return np.concatenate(bucket_images, axis=0), bucket_coords

def crop_image(img, x1, y1, x2, y2):
    return img[y2:y1, x2:x1]


def paths2pil(paths, resize=None, gray=None):
    raise NotImplemented

def paths2numpy(paths, resize=None, gray=None):
    raise NotImplemented

def pil2numpy(images):
    raise NotImplemented


def change_color_RGBA2RGB(np_images):
    raise NotImplemented

def resize_images(images, resize):
    raise NotImplemented

def glob_all_files(folder):
    return glob(os.path.join(folder, '*'))


def draw_rectangle(image, x1y1, x2y2, color, thickness):
    raise NotImplemented
    return img


def random_patch(background, target):
    raise NotImplemented
    return background


def draw_rectangles(image, coords, line_color, line_width):
    """
    Description:
        이미지에 사각형을 그립니다.

    :param image: Numpy, HWC
    :param coords: List, [x1, y1, x2, y2]
    :param line_color: tuple, iterable, (R, G, B)
    :param line_width: int
    :return:
    """

    raise NotImplemented
    return image

