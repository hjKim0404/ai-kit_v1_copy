import cv2
import numpy as np
from helper import draw_rectangle


def template_matching(background, target, threshold, blur_mode='blur', blur_kernel=(9, 9)):
    """
    :param background:
    :param target:
    :param threshold:
    :param blur_mode:
    :param blur_kernel:

    :return:

    :reference:
        https://opencv-python.readthedocs.io/en/latest/doc/11.imageSmoothing/imageSmoothing.html
    """

    ori_background = background.copy()
    target_h, target_w = target.shape
    mode = "cv2.TM_CCOEFF_NORMED"

    # Blur
    if blur_mode.lower() == 'blur':
        background = cv2.blur(background, blur_kernel, 1)
        target = cv2.blur(target, blur_kernel, 1)

    elif blur_mode.lower() == 'gaussian':
        background = cv2.GaussianBlur(background, blur_kernel, 1)
        target = cv2.GaussianBlur(target, blur_kernel, 1)

    else:
        raise NotImplementedError

    # Matching
    res = cv2.matchTemplate(background, target, eval(mode))

    # Left Top Coordinates
    x1y1 = np.stack((np.where(res > threshold)[1], np.where(res > threshold)[0]), axis=-1)

    # Right bottum Coordinates
    x2y2 = x1y1 + np.array([target_w, target_h])

    # coordinate
    coordinates = np.concatenate([x1y1, x2y2], axis=1)

    # Draw images
    img_with_rect = draw_rectangle(ori_background, x1y1, x2y2, (255, 0, 0), thickness=1)

    return res, img_with_rect, coordinates


if __name__ == '__main__':
    ## matching case 1
    # img = cv2.imread('/Users/seongjungkim/PycharmProjects/Wally/images/full_images/1_3.jpg', 0)
    # template = cv2.imread('/Users/seongjungkim/PycharmProjects/Wally/images/wally_face_tight_crop/2_1.png', 0)
    # res, img = template_matching(img, template, 0.85)

    # Matching case 2
    # Compare wally to wally
    # face_folder = '/Users/seongjungkim/PycharmProjects/Wally/images/wally_face_tight_crop'
    # paths = glob_all_files(face_folder)
    # imgs = paths2numpy(paths, gray=True)
    #
    # #
    # background = imgs[4]
    # background = zero_padding(background, 70, 70)
    #
    # #
    # target = imgs[3]
    # res, img_with_rect = template_matching(background, target, 0.80)
    #
    # #
    # show_images([img_with_rect, res, background, target])

    pass




