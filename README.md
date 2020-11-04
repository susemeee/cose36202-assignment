
# 2020-2R 기계학습 Term Project

2014210040 이수호

## 1. 코드에 대한 설명

코드는 크게 다음 세 파일로 분리되어 있습니다.

1. main.py
2. classifier.py
3. vectorizer.py

### 1-1. main.py

main.py에서는 크게 다음과 같은 역할을 수행합니다.

1. csv 파일을 읽어서 pandas.DataFrame으로 변경 (`load_data()` 함수가 수행)
2. DataFrame 내의 data(문장)를 전처리 (`preprocess()` 함수가 수행)
3. 모델 학습 (`train()` 함수가 수행)
4. 모델 테스트 (`test()` 함수가 수행)

### 1-2. classifer와 vectorizer

classifier는 어떤 feature vector에 대한 classification 작업을 하는 모델입니다(과제에서는 *ham*과 *spam*을 구별). classifier의 input은 보통 feature vector이기 때문에, 자연어 처리 모델을 만들 떄 보통 vectorizer를 함께 사용합니다. vectorizer는 자연어로 이루어진 문장에서 feature extraction을 하여 vector 포맷으로 바꿔줍니다. classifier와 vectorizer는 모델의 중요한 구성 요소이기 때문에, 어떤 classifier와 vectorizer를 사용하고 있는지를 쉽게 확인할 수 있도록 두 모델을 리턴하는 함수를 별도의 파이썬 파일로 분리하였습니다.

classifier와 vectorizer 파일에는 각각 `get_classifier()`와 `get_vectorizer()` 함수가 정의되어 있습니다. 각각 함수에서는 main.py에서 사용하는 classifier와 vectorizer를 리턴하여 줍니다.

classifier에는 모델 트레이닝을 위한 `classifier#fit()` 함수와 input에 대해 output을 예측하는 `classifier#predict()` 함수가 정의되어 있어야 합니다.

vectorizer에는 모델 트레이닝을 위한 `vectorizer#fit()` 함수와 input string에 대해 output vector를 반환하는 `vectorizer#transform()` 함수가 정의되어 있어야 합니다.

scikit-learn에서 제공하는 classifier와 vectorizer 구현체에는 공통적으로 해당 함수가 정의되어 있어, 해당 구현체를 사용한다면 추가적으로 코드를 수정할 부분은 없습니다. 다만, classifier 또는 vectorizer를 직접 구현하여 사용할 경우에, 각 함수에서 리턴하는 인스턴스에는 해당 메서드가 구현이 되어있어야 합니다.



## 2. 문제에 대한 접근법

### 2-1. Bag of Words


### 2-2. n-gram


### 2-3. 적절한 Vectorizer 선택하기


### 2-4. 적절한 Classifier 선택하기


## 3. 코드 실행 및 환경 설정법

### 3-1. 환경 설정하기

- Python 런타임으로는 Anaconda 2020-07(Python 3.8)을 사용하였습니다. OS에 따른 종속성은 없습니다.
- 혹시 몰라
  ```shell
  pip freeze > requirements.txt
  ```
  로 현재 사용중인 디펜던시를 pinning 해놓았습니다. 이 경우
  ```shell
  pip install -r requirements.txt
  ```
  를 실행하여 디펜던시를 설치하면 됩니다.

### 3-2. 코드 실행하기

1. data 폴더 내에 train data와 test data가 있어야 합니다.
  - train data와 test data의 csv 형식은 Kaggle competition에서 주어진 csv 파일의 형식과 같습니다.
  - train data의 이름은 `train.csv`고, test data의 이름은 `leaderboard_test_file.csv`입니다.

2. train data와 test data를 폴더 내에 두었다면,
  ```shell
  python main.py
  ```

  를 호출하여 코드를 실행할 수 있습니다.
