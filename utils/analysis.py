import numpy as np


def image_info(images):
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