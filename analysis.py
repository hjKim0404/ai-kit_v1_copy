from helper import paths2numpy, glob_all_files, get_names, zero_padding, draw_rectangle, show_images
from template_matching import  template_matching
import matplotlib.pyplot as plt
import numpy as np
import cv2


def image_info(images):
    sizes = []
    for image in images:
        # image shape H, W ,C
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


if __name__ == '__main__':
    # 아래 코드를 지우지 마세요 #

    # load face images
    full_folder = '/Users/seongjungkim/PycharmProjects/Wally/images/full_images_1'
    paths = glob_all_files(full_folder)
    full_imgs = paths2numpy(paths, gray=True)

    # load face images
    face_folder = '/Users/seongjungkim/PycharmProjects/Wally/images/wally_face_tight_crop'
    paths = glob_all_files(face_folder)
    imgs = paths2numpy(paths, gray=True)

    # extract name from paths
    names = get_names(paths)

    bucket_hw = []
    for ind, img in enumerate(imgs):
        h, w = img.shape
        name = names[ind]
        bucket_hw.append([h, w])

    # 최대 최소 H, W, ratio 을 구해서 출력한다.
    image_info(imgs)

    # original 이미지를 생성합니다.
    pad_imgs = []
    for img in imgs:
        pad_imgs.append(zero_padding(img, 60, 60))

    # padding 된 이미지를 생성합니다.
    for h, w in bucket_hw:
        imgs = paths2numpy(paths, gray=True, resize=(w, h))
        for img in imgs:
            pad_imgs.append(zero_padding(img, 60, 60))

    # 검사지 생산 - multiple image to single image
    # 적절한 이미지 갯수를 적어 놓으세요.
    # res_imgs = np.reshape(np.array(pad_imgs), (6, 5, 60, 60)).transpose([0, 2, 1, 3]).reshape(6*60, 5*60)

    # targets 이미지를 생성합니다.
    target_imgs = []
    for h,w in bucket_hw:
        for img in imgs:
            target_imgs.append(cv2.resize(img, dsize=(w, h)))


    # 모든 patch 을 활용해 검사지를 확인합니다.
    # for target in target_imgs:
    #     _, image_with_rect, rect = template_matching(res_imgs, target, 0.6, blur_kernel=(5, 5))
    #     show_images([image_with_rect, target])

    #
    # # matching all images
    # full_with_rect = []
    # for full_img in full_imgs[:]:
    #     bucket_rect = []
    #     for target in target_imgs[-1:]:
    #         _, image_with_rect, rect = template_matching(full_img, target, 0.8, blur_kernel=(5, 5))
    #         # add all rect
    #         bucket_rect.append(rect)
    #     bucket_rect = np.concatenate(bucket_rect, axis=0)
    #     full_with_rect.append(draw_rectangle(full_img, bucket_rect[:, :2], bucket_rect[:, 2:], (255, 0, 0), 1))
    #
    # for full_img in full_with_rect:
    #     plt.figure(figsize=(13,13))
    #     plt.imshow(full_img)
    #     plt.show()
