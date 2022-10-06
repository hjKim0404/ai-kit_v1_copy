import numpy as np
from tensorflow.keras.utils import Sequence
from helper import glob_all_files, random_patch, show_images, paths2numpy, images_cropper
from helper import face_resize_augmentation
from helper import image_info, random_imaug


class TightFaceProvider(Sequence):
    """
    해당 클래스는 월리 얼굴을 Tight 하게 잘라 만든 데이터를 활용하기 위해 만들어진 DataGenerator 입니다.
    # Foreground 생성
        1. tight 이미지를 이미지내 모든 h, w에 대해 resize 합니다.
    # Background 생성

    Train Batch
        background 와 foreground 가 반반 들어갑니다.
    """
    def __init__(self, fg_folder, bg_folder, val_fg_folder, val_bg_folder, batch_size):
        """
        :param fg_folder: Tight 하게 Crop 된 월리 이미지들이 들어 있는 폴더
        :param bg_folder: 월리가 있는 부분은 block 처리된 full image 가 들어 있는 폴더
        """
        # 지정된 배치 사이즈
        self.batch_size = batch_size

        # 학습용, 검증용 월리 이미지를 가져온다.
        self.fg_imgs = self.set_fg_imgs(fg_folder)
        self.val_fg_imgs = self.set_fg_imgs(val_fg_folder)
        
        # 월리 얼굴의 최대 높이, 최대 너비를 구한다.
        (max_h, min_h), (max_w, _), (_, _) = image_info(self.fg_imgs + self.val_fg_imgs)
        # 높이와 너비 중 더 큰 것을 max_length에 저장한다.
        self.max_length = np.maximum(max_h, max_w)
        # stride 를 설정한다.
        self.stride = int(self.max_length/4)

        # background 이미지 폴더 경로에 저장된 이미지들을 가져온다.
        self.bg_imgs = self.set_bg_imgs(bg_folder)
        self.val_bg_imgs = self.set_bg_imgs(val_bg_folder)

        # background 이미지를 무작위로 섞습니다.
        np.random.shuffle(self.bg_imgs)
        np.random.shuffle(self.val_bg_imgs)
        
    def __len__(self):
        """
        한 epoch 당 step 수를 구해주기 위한 매직 메소드.
        한 epoch 당 step 수: 전체 데이터 수  / 배치 크기
        """
        # 배치 사이즈가 절반인 이유는 step 수를 맞추기 위함
        return np.int(np.ceil(len(self.bg_imgs)/(self.batch_size/2)))

    def __getitem__(self, index):
        """
        인스턴스를 인덱싱 했을 때, 인덱스에 해당하는 이미지들과 라벨들을 리턴하는 매직 메소드.
        """
        # 분할된 background 이미지의 인덱싱 범위를 지정한다.
        start = index * int(self.batch_size/2)
        end = (index+1) * int(self.batch_size/2)

        # background 인덱싱 후 길이에 맞게 라벨을 붙인다.
        # background label: 0
        bg_imgs = self.bg_imgs[start: end]
        
        batch_xs, batch_ys = self.generate_pair_imgs(bg_imgs, self.fg_imgs)
        
        return batch_xs, batch_ys
    
    
    def get_validation_set(self):
        """
        검증용 이미지 셋을 불러오기 위한 메소드입니다.
        해당 클래스의 self.val_bg_imgs, self.val_fg_imgs 변수를 통해 불러옵니다.
        """
        bg_imgs = self.val_bg_imgs
        fg_imgs = self.val_fg_imgs
        
        batch_xs, batch_ys = self.generate_pair_imgs(bg_imgs, fg_imgs)
        
        return batch_xs, batch_ys
    
    
    def on_epoch_end(self):
        """
        에포크 종료 시 마다 bg 이미지들을 섞어줍니다.
        """
        np.random.shuffle(self.bg_imgs)
        np.random.shuffle(self.val_bg_imgs)

    def set_fg_imgs(self, path):
        """
        월리 이미지를 가져옵니다.
        """
         # 월리 이미지들이 저장된 경로를 가져온다.
        fg_imgs = glob_all_files(path)
        # 가져온 경로에서 이미지 파일을 numpy 배열로 바꾼다.
        fg_imgs = paths2numpy(fg_imgs)
        # numpy 배열의 월리 얼굴들을 사이즈별로 새로 생성한다.
#         fg_imgs = face_resize_augmentation(fg_imgs)
        
        return fg_imgs
    
    def set_bg_imgs(self, path):
        """
        배경 이미지를 crop 해서 가져옵니다.
        """
        # background 이미지의 경로를 가져온다.
        bg_imgs = glob_all_files(path)
        # 가져온 경로에서 이미지 파일을 numpy로 바꾼다.
        bg_imgs = paths2numpy(bg_imgs)
        # 크기가 max_length인 filter를 siride 단위로 background 이미지를 분할한다.
        bg_imgs, _ = images_cropper(bg_imgs, self.stride, self.stride, self.max_length+1, self.max_length+1)
        
        return bg_imgs
    
    def generate_pair_imgs(self, bg_imgs, fg_imgs):
        # background 인덱싱 후 길이에 맞게 라벨을 붙인다.
        # background label: 0
        bg_imgs = random_imaug(bg_imgs, is_bg=True)
        bg_labs = np.zeros(len(bg_imgs))

        # foreground 이미지에 해당하는 배치 사이즈를 지정한다.
        fg_batch = len(bg_imgs)
        # 랜덤하게 월리 얼굴을 가져오기 위한 fg_batch 개수 만큼의 인덱스를 무작위로 가져온다.
        indices = np.random.choice(np.arange(len(fg_imgs)), size=fg_batch)
        # 위에서 지정한 랜덤 인덱스에 맞는 월리 얼굴을 가져온다.
        fg_imgs = [fg_imgs[ind] for ind in indices]
        
        # foreground 이미지에 해당하는 배치 사이즈를 지정한다.
        fg_batch = len(bg_imgs)
        fg_imgs = random_imaug(fg_imgs)
        
        # 길이에 맞게 라벨을 붙인다.
        # foreground label: 1
        fg_labs = np.ones(len(fg_imgs))

        # 기존의 월리 얼굴을 background의 랜덤한 위치에 붙여준 후, 그 이미지를 foreground 이미지로 다시 저장한다.
        for ind, img in enumerate(bg_imgs.copy()):
            fg_imgs[ind] = random_patch(img, fg_imgs[ind])

        # batch_xs에는 background, foreground의 이미지들이 들어간다.
        batch_xs = np.concatenate([fg_imgs, bg_imgs], axis=0)
        
        # batch_ys에는 background, foreground의 라벨들이 들어간다.
        batch_ys = np.concatenate([fg_labs, bg_labs], axis=0)
        
        return batch_xs, batch_ys
    
    
if __name__ == '__main__':
    train_fg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/wally_face_tight_crop'
    train_bg_folder = '/Users/pai/Downloads/Find_Wally_Deploy/data/block_imgs'
    tfp = TightFaceProvider(train_fg_folder, train_bg_folder, 64)
    show_images(tfp[0][0])
