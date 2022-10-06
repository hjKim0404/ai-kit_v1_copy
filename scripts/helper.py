import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import imgaug as ia
import imgaug.augmenters as iaa


def show_images(images, figsize=(10, 10), titles=None):
    """
    Describes:
        imgs 로 들어오는 np.array 형태의 이미지들을 화면에 출력합니다.
    Parameter:
        imgs : np.array, 이미지 배열이 들어옵니다.
        titles   : list, 각 이미지의 제목을 순서대로 넣음
        figsize : tuple, 출력될 이미지의 크기를 지정함 (이미지의 너비, 이미지의 높이)
    """
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
            pass

    plt.tight_layout()
    plt.show()


def cropper(img, stride_h=10, stride_w=10, filter_h=35, filter_w=35):
    """
    Describes:
        background 이미지를 stride 단위마다 filter 의 h,w 크기로 분할해주는 함수입니다.
    Parameter:
        img : np.array, 분할되어질 background 이미지 1개
        stride_h : int, filter 가 이동할 높이
        stride_w : int, filter 가 이동할 너비
        filter_h : int, filter 의 높이
        filter_w : int, filter 의 너비
    Return
        tuple[0] : np.array, 하나의 이미지를 분할하여 나온 이미지 조각 (N : 분할 이미지 개수, H: 높이, W: 너비, D: 컬러채널)
        tuple[1] : list, 각 이미지 조각들의 좌표
    """
    # 가져온 background의 크기와 범위를 구한다.
    img_h, img_w = img.shape[:2]

    # 범위를 지정할 때 전체 이미지 크기에서 filter 크기만큼 빼는 이유: filter가 background를 넘어갈 수 있기 때문
    range_h = range(0, img_h - filter_h, stride_h)
    range_w = range(0, img_w - filter_w, stride_w)

    # 지정된 단위마다 crop을 진행하여 주는 부분.
    # crop의 shape가 지정한 filter와 일치할 경우에만 crop된 이미지와 좌표를 append 한다.
    crop_imgs = []
    crop_crds = []
    for y in range_h:
        for x in range_w:
            crop_img = img[y: y+filter_h, x: x+filter_w]
            if crop_img.shape[:2] == (filter_h, filter_w):
                crop_imgs.append(crop_img)
                crop_crds.append([x, y, x+filter_w, y+filter_h])

    return np.array(crop_imgs), crop_crds


def images_cropper(images, stride_h=10, stride_w=10, filter_h=35, filter_w=35):
    """
    Describes:
        여러 개의 이미지들 각각에 cropper 함수를 적용하여 잘라준 후 하나로 묶어줍니다.
    Parameter:
        imgs : list, 여러 개의 background 이미지가 있는 list 를 넣어줘야 합니다.
        stride_h : int, filter 가 이동할 높이
        stride_w : int, filter 가 이동할 너비
        filter_h : int, filter 의 높이
        filter_w : int, filter 의 너비
     Return:
        tuple[0] : np.array, 여러 개의 이미지를 분할하여 나온 이미지 조각 (N : 분할 이미지 개수, H: 높이, W: 너비, D: 컬러채널)
        tuple[1] : list, 각 이미지 조각들의 좌표
    """
    bucket_images = []
    bucket_coords = []
    for image in images:
        cropped_imgs, cropped_crds = \
        cropper(image, stride_h=stride_h, stride_w=stride_w, filter_h=filter_h, filter_w=filter_w)
        bucket_images.append(cropped_imgs)
        bucket_coords.extend(cropped_crds)

    return np.concatenate(bucket_images, axis=0), bucket_coords


def paths2pil(paths, resize=None, gray=None):
    """
    Describes:
        paths 에 있는 이미지 파일들을 pil 형태로 list 에 저장합니다.
    Parameter:
        paths : str, 해당 경로의 이미지 파일들을 불러옵니다.
        gray : none,
            1. gray 매개변수에 값을 넣을 경우, 경로의 이미지들을 흑백사진으로 변환하여 저장합니다.
            2. gray 매개변수에 값을 넣지 않을 경우, 경로의 이미지들을 컬러사진으로 변환하여 저장합니다.
        resize : tuple, (너비, 높이) 형식으로 설정되어있을 경우, 해당 값으로 이미지들의 크기를 변환합니다.
    Return:
        list : pil 타입의 이미지들이 list 에 담겨 반환됩니다.
    """
    bucket_pils = []
    for path in paths:
        if gray:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')

        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        bucket_pils.append(img)

    return bucket_pils


def paths2numpy(paths, resize=None, gray=None):
    """
    Describes:
        경로에 있는 이미지 파일들을 paths2pil 함수로 pil 타입으로 바꾸고 다시 np.array 형태로 바꿔줍니다.
    Parameter:
        paths : str, 해당 경로의 그림파일을 불러옴
        gray : none,
            1. gray 매개변수에 값을 넣을 경우, 경로의 이미지들을 흑백사진으로 변환하여 저장합니다.
            2. gray 매개변수에 값을 넣지 않을 경우, 경로의 이미지들을 컬러사진으로 변환하여 저장합니다.
        resize : tuple, (너비, 높이) 형식으로 설정되어있을 경우, 해당 값으로 이미지들의 크기를 변환합니다.
    Return:
        list : np.array 타입의 이미지들이 list 에 담겨 반환됩니다.
    """
    pils = paths2pil(paths, resize=resize, gray=gray)
    return pil2numpy(pils)


def pil2numpy(images):
    """
    Describes:
        pil 타입의 이미지 들이 np.array 타입으로 변경됩니다.
    Parameter:
        images : list, pil 타입의 이미지가 담긴 list 가 들어옵니다.
    Return:
        list : np.array 형태의 이미지들이 list 에 담겨 반환됩니다.
    """
    return [np.array(image) for image in images]


def glob_all_files(folder):
    """
    Describes:
        folder 경로에 있는 모든 파일의 경로들을 list 로 저장합니다.
    Parameter:
        folder : str, folder 경로가 들어옵니다.
    Return:
        list : 지정한 경로 안에 있는 모든 파일의 경로들을 list 로 반환합니다.
    """
    return glob(os.path.join(folder, '*'))


def random_patch(bg_img, fg_img):
    """
    Describes:
        background 이미지의 임의의 위치에 target 이미지(월리 얼굴)를 붙여줍니다. 해당 이미지는 foreground 이미지가 됩니다.
    Parameter:
        bg_img : np.array, 배경 백그라운드 이미지
        fg_img : np.array, 월리 얼굴 포어그라운드 이미지
    Return:
        np.array : background 이미지에 target 이미지가 붙은 np.array 형식의 이미지가 리턴됩니다.
    """
    # background와 foreground의 h, w 불러온다.
    bg_h, bg_w = bg_img.shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    # 해당 조건에 맞는 경우 다음 코드를 실행한다.
    assert (bg_h >= fg_h) & (bg_w >= fg_w), print("{}{}{}{}".format(bg_h, fg_h, bg_w, fg_w))

    # fg_img가 들어갈 위치의 범위를 지정한다.
    range_h = range(0, bg_h - fg_h)
    range_w = range(0, bg_w - fg_w)

    # fg_img를 붙일 임의의 좌표를 선정한다.
    rand_h = np.random.choice(range_h, 1)[0]
    rand_w = np.random.choice(range_w, 1)[0]

    bg_img[rand_h: rand_h+fg_h, rand_w: rand_w+fg_w] = fg_img
    
    return bg_img


def random_imaug(images, is_bg = False):
    """
    Describes:
        이미지들에 여러가지 효과들을 적용시킨 후 저장하는 함수입니다.
        참고: https://github.com/aleju/imgaug (패키지 깃헙)
             https://imgaug.readthedocs.io/en/latest/source/examples_basics.html (코드)
    Parameter:
        fg_imgs : list, 포그라운드 이미지들
    Return:
        list : 여러가지 효과가 적용된 포그라운드 이미지
    """
    ia.seed(1)
    # Example batch of images.
    
    # bg 이미지들에 적용시키는 augmentation 들
    if is_bg:
        seq = iaa.Sequential([
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Sometimes(
                0.5,
                iaa.Rotate((-10, 10)),
                iaa.MultiplyBrightness((0.5, 1.5)),
            ),
            iaa.Fliplr(0.5),
#             iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5), per_channel=True)
        ], random_order=True)  # apply augmenters in random order

    # fg 이미지들에 적용시키는 augmentation 들
    else:
        seq = iaa.Sequential([
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Sometimes(
                0.5,
                iaa.Rotate((-10, 10)),
                iaa.MultiplyBrightness((0.5, 1.5)),
            ),
            iaa.Resize((0.9, 1.0))
        ], random_order=True)  # apply augmenters in random order
        
    images_aug = seq(images=images)
        
    return images_aug


def draw_rectangles(img, coords, color, width, predicts):
    """
    Describes:
        img 의 coords 좌표에 color 색깔과 width 선 굵기를 가진 사각형을 그려줍니다.
    Parameter:
        img : np.array, 원본 background 이미지
        coords : np.array, 월리 얼굴이 있을 것으로 예상되는 좌표들 [x1,y1,x2,y2]
        color : tuple, (R,G,B) 선의 색깔.
        width : int, 선의 두께
    Return:
        np.array : 목표 지점에 사각형이 그려진 이미지를 반환합니다.
    """
    filtered_coords = rectangle_filter(coords, 0.3, predicts)

    for coord in filtered_coords:
        img = cv2.rectangle(img, tuple(coord[:2]), tuple(coord[2:]), color, width)

    return img


def rectangle_filter(coords, overlapThresh, predicts):
    """
    Describes:
        NMS 기법을 활용하여 테스트 과정에서 한 오브젝트에 사각형 여러 개가 그려지는 것을 방지합니다.
        참고: https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    Parameter:
        coords : np.array, 월리 얼굴이 있을 것으로 예상되는 좌표들 [x1, y1, x2, y2]
        overlapThresh : float, 중첩 비율 기준 값
    Return:
        np.array, 중첩 비율 기준치를 초과하는 사각형들을 제외한 나머지 사각형들의 좌표가 저장된 넘파이 배열
    """

    # 사각형 좌표가 비어있을 때는 빈 리스트를 리턴합니다.
    if len(coords) == 0:
        return []

    # 사각형 좌표들 중 이미지에 그려질 예정인 사각형들의 좌표만을 저장하는 리스트 입니다.
    pick = []

    # 사각형 좌표의 값들 각각을 별도의 리스트로 저장해줍니다.
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    # 각 영역의 사각형 넓이를 계산합니다.
    area = (x2 - x1) * (y2 - y1)

    # 예측값들을 argsort 하여 인덱스화 합니다.
    pred_idxs = np.argsort(predicts)

    # 예측값 인덱스 리스트의 값들이 전부 사라질 때까지 반복문을 돌립니다.
    while len(pred_idxs) > 0:
        # 예측값 인덱스의 마지막 자리에 있는 값(가장 예측치가 높은 값)을 pick 리스트에 넣어준 후 서프레션 적용 대상 리스트에 저장합니다.
        last = len(pred_idxs) - 1
        i = pred_idxs[last]
        pick.append(i)
        suppress = [last]  # pred_idxs에 있는 애들 없애는 역할

        # 중첩 비율 계산을 통해 필터링할 사각형들을 찾는 과정입니다.
        # 예측값 인덱스들을 모두 순회합니다.
        for pos in range(0, last):

            # 현재 순회하는 인덱스 값을 저장
            j = pred_idxs[pos]

            # 현재 인덱스 좌표와 마지막 위치에 있는 인덱스 좌표의 x1y1의 최대값, x2y2의 최소값을 구합니다.
            # 이는, 두 사각형의 겹친 부분에서 생기는 사각형의 좌표가 됩니다.
            # 겹쳐있지 않을 경우, 두 사각형과 겹치지않는 별개의 사각형이 생기게 됩니다.
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # 위에서 구한 값을 토대로, 새로 생긴 사각형의 너비와 높이를 구합니다.
            # 두 사각형이 겹쳐있을 경우에는 양의 값이 w and h 에 저장됩니다.
            # 겹쳐있지 않을 경우, 0 이 w or h에 저장됩니다.
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            # 두 사각형이 겹쳐진 곳의 넓이와 현재 인덱스에 해당하는 사각형의 넓이의 중첩 비율을 구합니다.
            # 겹쳐있지 않을 경우, w or h 의 값이 0이기 때문에 중첩 비율 또한 0이 됩니다.
            overlap = float(w*h)/area[j]

            # 중첩 비율이 지정해준 기준치 이상일 경우, 해당 사각형을 제외 리스트에 추가합니다.
            if overlap > overlapThresh:
                suppress.append(pos)

        # pred_idxs 에서 suppress 리스트에 들어간 값들을 제거합니다. 이후 pred_idxs 에 값이 남아있지 않을 경우, 반복을 종료합니다.
        pred_idxs = np.delete(pred_idxs, suppress)

    return coords[pick]


def face_resize_augmentation(images):
    """

    해당 함수는 wally 의 얼굴을 tight 하게 crop 해놓은 이미지를 사용합니다.
    예를 들어
    아래와 같이 3가지 각기 다른 h, w 을 가진 image들이 있다면

            image 1    image 2      image 3
    h, w    (10, 6)     (10,7)      (10,8)

    각 이미지들을  각각의 h, w으로  resize 시킵니다.

            image 1    image 2      image 3
    h, w    (10, 6)     (10,6)      (10,6)
    h, w    (10, 7)     (10,7)      (10,7)
    h, w    (10, 7)     (10,8)      (10,8)

    :return:
    """
    # 월리 얼굴들의 모든 사이즈를 가져온다.
    sizes = []
    for image in images:
        sizes.append(image.shape[:2])

    # 각 이미지마다 위에서 구한 모든 사이즈들 별로 적용시켜 저장한다.
    resized_imgs = []
    for image in images:
        for h, w in sizes:
            resized_imgs.append(cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_NEAREST))
    return resized_imgs


def image_info(images):
    """
    Describes:
        인자로 받은 이미지들의 높이, 너비, 비율(너비/높이)들을 구해 각 항목의 최대, 최소 값을 리턴하는 함수입니다.
    Parameter:
        images, list, np.array 형식의 이미지들이 원소로 저장된 리스트가 들어갑니다.
    Return:
        tuple[0] : 이미지들의 높이 값 중 최대, 최소값이 리턴됩니다.
        tuple[1] : 이미지들의 너비 값 중 최대, 최소값이 리턴됩니다.
        tuple[2] : 이미지들의 비율 값 중 최대, 최소값이 리턴됩니다.
    """
    sizes = []
    for image in images:
        height, width = image.shape[:2]
        sizes.append([height, width])

    # 최대 최소 H, W, ratio 을 구해서 출력한다.
    sizes = np.array(sizes)
    max_h, max_w = sizes.max(axis=0)
    min_h, min_w = sizes.min(axis=0)
    max_ratio = (sizes[:, 1] / sizes[:, 0]).max()
    min_ratio = (sizes[:, 1] / sizes[:, 0]).min()

    print('Max Height : {} \t Min Height : {}'.format(max_h, min_h))
    print('Max Width : {} \t Min Width : {}'.format(max_w, min_w))
    print('Max W/H Ratio : {} \t Min Ratio : {}'.format(max_ratio, min_ratio))

    return (max_h, min_h), (max_w, min_w), (max_ratio, min_ratio)
