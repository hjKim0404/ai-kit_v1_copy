# AI-KIT, 월리를 찾아라

ver201204



### AI-KIT, 월리를 찾아라 란?

해당 **교육용 프로그램**에서는 **월리를 찾아라** 게임에서 월리가 어디있는 지를 스스로의 힘으로 찾아낼 수 있는 **인공지능**이 구현되어 있습니다.

이 프로그램은 **인공지능**을 배우는데 있어서 **필수적인 기술**들을 중심으로 구성되어 있습니다.

때문에, 해당 프로그램을 공부하는 것을 통하여 **인공지능 프로그램 개발**에 대한 **전반적인 이해**를 얻을 수 있을 것입니다.



### 사용 방법



### 특징

* Python 기반으로 제작
* Numpy, Tensorflow 등의 인공지능 개발에 필수적인 패키지 사용
* Keras, CNN(Convolutional Neural Network) 을 활용한 전문적인 인공지능 모델 학습 이론 탑재



### 구성

월리 프로그램을 다운로드 받을 경우, 다음과 같이 폴더가 구성되고 그 내용물들은 다음과 같습니다.

* **data : 인공지능 교육 및 검증, 평가 에 사용되는 리소스들이 저장된 폴더입니다.**
  
  * block_imgs : 월리를 찾아라의 백그라운드 이미지들이 저장된 폴더 입니다.
    
  * wally_face_tight_crop : 월리의 얼굴부분만으로 이루어진 포그라운드 이미지들이 저장된 폴더입니다.
    
  * full_images_val : 검증에 사용되는 월리 이미지가 저장된 폴더입니다.
    
  * full_images_test : 최종 테스트에 사용되는 월리 이미지가 저장된 폴더입니다.
  
  
  
* **models : 학습이 완료된 인공지능 모델이 저장되는 폴더입니다.**

  * best_model: 예제로 만들어진 인공지능 학습 모델이 저장된 폴더입니다.

  

* **scripts : 프로그램을 직접적으로 구동하기 위한 스크립트들이 저장된 폴더입니다.**

  * 모델학습하기: 인공지능의 학습 모델을 제작하기 위한 코드입니다. 

  * 모델불러오기: 학습이 완료된 인공지능 모델을 불러와 테스트하기 위한 코드입니다.

    

* **utils : 프로그램을 구성하는 핵심 기능들을 구현한 파일들이 저장된 폴더입니다.**

  * datagenerator : 리소스 이미지들을 교육용 데이터에 알맞은 형식으로 변환한 후 활용하기 위한 클래스가 정의된 파일입니다.
  * helper : 프로그램 내의 모든 함수들이 저장된 파일입니다.
                  함수들은 경로로부터 이미지 불러오기, 이미지 수정 등 다양한 용도로 사용됩니다.



### 변경점

* 월리 프로그램과 연관되는 analysis.py, augmentator.py 의 내용들 helper로 이동후 파일들 삭제

* Validation과 Test를 구분하기 위해 data폴더에 full_images_test 폴더를 추가하고, 
  full_images_val 에 있던 이미지 한 장(11번) 을 test 폴더로 이동.

* 파이참, 주피터에서 양측에서의 원활한 구동을 위해 model_training.py, model_training.ipynb 및 
  model_test.py, model_test.ipynb 두 종류씩 파일 생성

* helper.py에 rectangle_filter 함수를 추가하여 한 오브젝트에 사각형이 여러 개 그려지는 것을 방지. (NMS 기법 사용) 
  
* datagenerator.py의 74번 라인을 기존 방식에서 fg_batch = len(bg_imgs) 로 수정. (백그라운드 배치가 지정된 값보다 덜 들어올 경우 학습 과정 도중 에러가 발생했기 때문)
  
* 주석 추가

  

### TODO 

- [ ] datagenerator 을 data로 이름 바꾸기 

- [x] augmentation data 로 옮기기  *datagenerator 대신 helper로 이동 (함수는 함수끼리, 클래스는 클래스끼리 묶는 것이 좋아보였기 때문.)

- [x] analysis 지우고 helper 로 옮기기 

- [ ] package 로 만들어서 pip로 올리기 

- [ ] 교육 내용에 package 다운받는거 추가하기 

  

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
