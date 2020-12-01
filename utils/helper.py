import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


def show_images(images, figsize=(10, 10), titles=None):
    """
    이미지를 화면에 출력하는 함수입니다.
    """
    length = np.maximum(np.int(np.ceil(np.sqrt(len(images)))), 2)
    fig, axes = plt.subplots(length, length, figsize=figsize)
    axes = axes.ravel()
    for ind, ax in enumerate(axes):
        try:
            ax.imshow(images[ind])
            if titles:
                ax.set_title(titles[ind])
            ax.axis('off')
        except IndexError:
            pass

    plt.tight_layout()
    plt.show()


def cropper(img, stride_h=35, stride_w=35, filter_h=35, filter_w=35):
    """
    이미지를 일정 간격, 일정 크기로 crop 하는 함수입니다.
    """
    img_h, img_w = img.shape[:2]

    h_indices = range(0, img_h - filter_h, stride_h)
    w_indices = range(0, img_w - filter_w, stride_w)

    # 크롭된 이미지와 이미지의 좌표를 저장
    crop_imgs = []
    coordinates = []
    for y in h_indices:
        for x in w_indices:
            cropped_imgs = img[y: y+filter_h, x: x+filter_w]
            if cropped_imgs.shape[:2] == (filter_h, filter_w):
                crop_imgs.append(img[y: y+filter_h, x: x+filter_w])
                coordinates.append([x, y, x+filter_w, y+filter_h])

    return np.array(crop_imgs), coordinates


def images_cropper(images, stride_h=35, stride_w=35, filter_h=35, filter_w=35):
    bucket_images = []
    bucket_coords = []
    for image in images:
        cropped_imgs , cropped_crds = \
        cropper(image, stride_h=stride_h, stride_w=stride_w, filter_h=filter_h, filter_w=filter_w)
        bucket_images.append(cropped_imgs)
        bucket_coords.extend(cropped_crds)

    return np.concatenate(bucket_images, axis=0), bucket_coords


def paths2pil(paths, resize=None, gray=None):
    """
    경로를 이미지로 바꿔주는 함수입니다.
    """
    imgs = []
    for path in paths:
        if gray:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')

        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        imgs.append(img)

    return imgs


def paths2numpy(paths, resize=None, gray=None):
    """
    경로를 ndarray로 바꿔주는 함수입니다.
    """
    pils = paths2pil(paths, resize=resize, gray=gray)
    return pil2numpy(pils)


def pil2numpy(images):
    """
    이미지를 ndarray로 바꿔주는 함수입니다.
    """
    return [np.array(image) for image in images]


def glob_all_files(folder):
    """
    folder에 있는 모든 파일의 경로를 가져오는 함수입니다.
    """
    return glob(os.path.join(folder, '*'))


def random_patch(background, target):
    """
    background 이미지의 랜덤한 위치에 target 이미지를 붙이는 함수입니다.
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = target.shape[:2]

    range_h = range(0, bg_h - fg_h)
    range_w = range(0, bg_w - fg_w)

    # 범위 안에서 h, w 랜덤한 값 가져옴
    rand_h = np.random.choice(range_h)
    rand_w = np.random.choice(range_w)

    background[rand_h: rand_h+fg_h, rand_w: rand_w+fg_w] = target
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
    image = cv2.rectangle(image, tuple(coords[:2]), tuple(coords[2:]), line_color, line_width)
    return image

