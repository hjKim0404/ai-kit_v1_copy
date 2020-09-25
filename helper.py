import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


def show_count(i):
    msg = '\r {}'.format(i)
    sys.stdout.write(msg)
    sys.stdout.flush()


def save_images(save_folder, imgs):
    for ind, img in enumerate(imgs):
        show_count(ind)
        fname = os.path.join(save_folder, '{}.jpg'.format(ind))
        cv2.imwrite(fname, img)


def show_images(images, figsize=(10, 10), titles=None):

    length = np.maximum(int(np.ceil(np.sqrt(len(images)))), 2)
    fig, axes = plt.subplots(length, length, figsize=figsize)
    axes = axes.ravel()
    for ind, ax in enumerate(axes):
        try:
            ax.imshow(images[ind])
            if titles:
                ax.set_title(titles[ind])
            ax.axis('off')
        except IndexError:
            pass;
    plt.tight_layout()
    plt.show()


def get_name(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_names(paths):
    return [get_name(path) for path in paths]


def video2image(video_path, mode='rgb'):
    # load video
    cap = cv2.VideoCapture(video_path)

    # make folder
    name = os.path.split(video_path)[0]

    # divide
    count = 0
    imgs = []

    while(cap.isOpened()):
        show_count(count)

        # read 1 frame from video
        ret, frame = cap.read()

        # if video cannot extract to image, break while loop
        if ret == False:
            break

        if mode=='rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        count += 1

        # append image
        imgs.append(frame)

    # close stream
    cap.release()
    cv2.destroyAllWindows()
    return imgs


def zero_padding(image, max_h, max_w):
    """
    흑백이든 컬러든 상관없이 잘 작동합니다.
    확인 완료
    :param image: 3D Numpy , H W C
    """
    #
    img_h, img_w = image.shape[:2]

    max_h = np.maximum(img_h, max_h)
    h_gap = (max_h - img_h)
    h_left = int(h_gap/2)
    h_right = h_gap - h_left

    max_w = np.maximum(img_w, max_w)
    w_gap = (max_w - img_w)
    w_left = int(w_gap / 2)
    w_right = w_gap - w_left

    # 위에서 구한 index 로 어느 축별 padding 을 생성합니다.
    pad_each_axis = np.zeros([image.ndim, 2], dtype=np.int)
    pad_each_axis[0] = np.array([h_left, h_right])
    pad_each_axis[1] = np.array([w_left, w_right])

    # padding 을 합니다.
    pad_image = np.pad(image, pad_each_axis, mode='constant')
    return pad_image


def rect2square(np_images):
    """
    :param np_images: 4D Numpy , N H W C
    :return: square_np,  4D Numpy , N H W C
    """
    # 어디에다가 padding 을 해야 할지 결정합니다.
    h, w = np.shape(np_images)[1:3]
    if h > w:
        # axis 1이 더 크면 axis 2에다가 추가 패딩을 합니다.
        small = w
        large = h
        index = 2
    elif h < w:
        # axis 2이 더 크면 axis 1 에다가 추가 패딩을 합니다.
        small = h
        large = w
        index = 1
    else:
        # 크기가 같다면 padding 하지 않고 return 합니다.
        return np_images

    # 얼만큼 padding 을 해야 할지 결정합니다.
    gap = large - small
    pad_left = int(gap / 2)
    pad_right = gap - pad_left

    # 위에서 구한 index 로 어느 축별 padding 을 생성합니다.
    pad_each_axis = np.zeros([4, 2], dtype=np.int)
    pad_each_axis[index] = np.array([pad_left, pad_right])

    # padding 을 합니다.
    pad_images = np.pad(np_images, pad_each_axis, mode='constant')
    return pad_images


def cropper(img, stride_h=35, stride_w=35, filter_h=35, filter_w=35):

    h, w = img.shape[:2]

    # indices
    h_indices = range(0, h, stride_h)
    w_indices = range(0, w, stride_w)

    # crop images
    crop_imgs = []
    for h in h_indices:
        for w in w_indices:
            crop_img = img[h: h + filter_h, w: w + filter_w]
            if (crop_img.shape[:2]) == (filter_h, filter_w):
                crop_imgs.append(img[h: h + filter_h, w: w + filter_w])

    return np.array(crop_imgs)


def images_cropper(images, stride_h=35, stride_w=35, filter_h=35, filter_w=35):
    bucket_imgs = []
    for image in images:
        bucket_imgs.append(cropper(image, stride_h=stride_h, stride_w=stride_w, filter_h=filter_h, filter_w=filter_w))
    return np.concatenate(bucket_imgs, axis=0)


def crop_image(img, x1, y1, x2, y2):
    return img[y2:y1, x2:x1]


def paths2pil(paths, resize=None, gray=None):

    imgs = []
    for path in paths:
        img = None
        if gray:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')

        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        imgs.append(img)

    return imgs


def paths2numpy(paths, resize=None, gray=None):
    pils = paths2pil(paths, resize=resize, gray=gray)
    return pil2numpy(pils)


def pil2numpy(images):
    return [np.array(image) for image in images]


def change_color_RGBA2RGB(np_images):
    ret_imgs = []
    for image in np_images:
        ret_imgs.append(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
    return ret_imgs


def resize_images(images, resize):
    imgs = []
    for image in images:
        res_imgs = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_AREA)
        imgs.append(res_imgs)
    return imgs


def glob_all_files(folder):
    return  glob(os.path.join(folder, '*'))


def draw_rectangle(image, x1y1, x2y2,color, thickness):
    img = image.copy()
    n_sample = len(x1y1)
    for ind in (range(n_sample)):
        cv2.rectangle(img,  tuple(x1y1[ind]), tuple(x2y2[ind]), color, thickness)
    return img


def random_patch(background, target):
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = target.shape[:2]
    assert (bg_h >= fg_h) & (bg_w >= fg_w), print('{}{}{}{}'.format(bg_h , fg_h, bg_w , fg_w))

    # 가능한 좌표를 생성한다.
    range_h = np.arange(0, bg_h - fg_h)
    range_w = np.arange(0, bg_w - fg_w)

    # random 좌표 생성
    rand_h = np.random.choice(range_h, 1)[0]
    rand_w = np.random.choice(range_w, 1)[0]

    background[rand_h: rand_h + fg_h, rand_w: rand_w + fg_w] = target

    return  background

