# AI-KIT, 월리를 찾아라



<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#directory">Directory</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#todo">Todo</a></li>
    <li><a href="#previous-version">Previous Version</a></li>
  </ol>
</details>





<!-- ABOUT THE PROJECT -->

## About The Project

해당 교육용 프로그램에는 수많은 객체들 속에서 찾고자 하는 타겟을 찾아낼 수 있는 인공지능이 구현되어 있습니다. 이 프로그램은 인공지능을 배우는데 있어서 필수적인 기술들을 중심으로 구성되어 있습니다. 따라서 해당 프로그램 커리큘럼을 따라 학습한다면 인공지능 프로그램 개발에 대한 전반적인 이해를 얻을 수 있습니다.

<!-- 대표 이미지 추가-->

### Built With

* <a href ='https://www.python.org'>Python</a>기반 프로그래밍
* <a href='https://numpy.org'>Numpy</a>, <a href='https://www.tensorflow.org'>Tensorflow</a> 등의 인공지능 개발에 필수적인 패키지 사용
* <a href='https://keras.io'>Keras</a>, <a href='https://en.wikipedia.org/wiki/CNN'>CNN</a> 을 활용한 전문적인 인공지능 모델 학습 이론 탑재



<a name='directory'><!-- DIRECTORY --></a>

## Directory

```
├── README.md                           - README 파일
│
├── data/                               - 학습, 검증, 평가에 사용되는 리소스
│   ├── book1/                          - 월리를 찾아라 1권 책의 이미지
│   │   ├── common/                     - 네 개의 타켓에 대해 공통으로 사용되는 리소스
│   │   │   ├── block_imgs/             - 타겟 얼굴이 가려진 백그라운드 이미지
│   │   │   └── full_images_test/	- 평가에 사용되는 이미지
│   │   │ 
│   │   ├── wally/                      - 월리 리소스
│   │   │   ├── face_tight_crop/	- crop된 타겟 얼굴 이미지
│   │   │   └── full_images_val/	- 검증에 사용되는 이미지
│   │   │       ├── background/		- crop된 background 이미지
│   │   │       └── full_image/		- background 이미지
│   │   │
│   │   ├── girlfriend/                 - 월리 여자친구 리소스 (하위 구조 wally/와 동일)
│   │   │
│   │   ├── fake/                       - 가짜 월리 리소스 (하위 구조 wally/와 동일)
│   │   │
│   │   └── magician/                   - 마법사 리소스 (하위 구조 wally/와 동일)
│   │
│   ├── book2/                          - 월리를 찾아라 2권 책의 이미지 (하위 구조 /book1과 동일)
│   ...                                   ...
│   └── book6/                          - 월리를 찾아라 6권 책의 이미지 (하위 구조 /book1과 동일)
│ 
├── models/                   		- 학습이 완료된 인공지능 모델 저장
│   └── new_model/                      - 예제로 만들어진 인공지능 학습 모델
│ 
├── scripts/                 	       - 프로그램을 직접 구동하기 위한 스트립트
│   ├── model_training.py     		- 인공지능의 학습 모델을 제작 및 검증하기 위한 python 파일
│   └── model_test.py         		- 학습이 완료된 인공지능 모델을 불러와 테스트하기 위한 python파일
│   ├── model_training.ipynb  		- 인공지능의 학습 모델을 제작 및 검증하기 위한 ipynb 파일 #삭제
│   └── model_test.ipynb      		- 학습이 완료된 인공지능 모델을 불러와 테스트하기 위한 ipynb파일 #삭제
│ 
├── utils/                    		- 프로그램을 구성하는 핵심 기능
│   ├── datagenerator.py      		- 리소스 이미지들을 학습 데이터에 알맞은 형식으로 변환 후 활용하기 위한 클래스가 정의된 파일
│   └── helper.py             		- 프로그램 내의 모든 함수들이 정의된 파일
│ 
└── main.py
```



<!-- USAGE -->

## Usage

### 패키지 없이 타겟 찾기

1. <a href='https://studyai.co.kr/courses/ai-kit-영상-강의/'>AI-KIT 온라인 강의</a>를 4장까지 진행하여 교육용 리소스들을 만듭니다.
2. 강의 내용과 동일한 폴더 구조를 만듭니다.
3. 타겟 얼굴을 지운 배경이미지들을 data/block_imgs 폴더에, crop된 타겟 얼굴 이미지들을 data/face_tight_crop 폴더에 저장합니다. 
4. 5장 강의들을 공부하며 예제 코드를 따라 입력합니다.
5. 입력한 코드로 인공지능 모델을 제작합니다.
6. 직접 만든 인공지능 모델을 사용하여 월리를 찾아봅니다.

### 패키지 사용해서 타겟 찾기

```
이후 패키지 만들면 추가
```



<!-- TODO -->

## TODO

- [x] augmentation data 로 옮기기  *datagenerator 대신 helper로 이동 (함수는 함수끼리, 클래스는 클래스끼리 묶기 위함)
- [x] analysis 지우고 helper 로 옮기기 
- [x] directory 구조 변경
- [ ] datagenerator 을 data로 이름 바꾸기
- [ ] package 로 만들어서 pip로 올리기
- [ ] 교육 내용에 package 다운받는 과정 추가




<a name='previous-version'></a>

## ↓ Previous Version

### 사진

해당 Course 에서 아래 내용을 배울수 있습니다. 

1. AI, Machine Learning , Deep Learning 개념 학습 
2. Regression 개념과 Classification 개념 
3. Scikit Learn을 활용해 머신러닝 구현 (Regression, Classification)
4. Tensorflow( ver2.x ) 을 활용해 Deep Neural Network 으로 이미지 분류기 학습해보기 
5. Tensorflow 활용해 Convolution Neural Netowrk (이하 CNN) 으로 학습해보기
6. Tensorflow 을 활용해 Wally Data 학습 시키기
7. 학습된 모델을 라즈베리 파이에 이식하기 
8. 라즈베리 파이에서 텐서플로우 모델 수행해보기





<hr>

Copyright(c) 2020 by Public AI. All rights reserved.