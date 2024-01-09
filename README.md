딥러닝을 활용한 게의 나이 예측

## 🖥️ 프로젝트 소개
Kaggle에 존재하는 게 데이터를 사용하여 게의 나이를 예측
<br>

## 🕰️ 개발 기간
* 24.01.03일 ~ 24.01.09일 ※ 1차 완료

### ⚙️ 개발 환경
- `python`
- **IDE** : Pycharm

## 📌 데이터 분석
#### 데이터 확인 및 학습과 평가 데이터 분리
- 데이터의 크기는 3893개의 데이터
- 데이터의 변수는 성별, 게의 길이, 게의 직경(??), 높이, 무게, 껍질을 제외한 무게, 내장 무게, 껍질 무게, 나이로 구성
- 일반적으로 성별과 나이는 무관하기 떄문에 성별은 제외(domain knowledge)
- 나머지 데이터들은 sklearn을 활용하여 학습과 평가 데이터 분리 (8:2)

#### 상관관계 분석
- 독립변수와 종속변수 or 독립변수간의 상관관계를 보기 위해 상관관계 분석 진행
- 독립변수 대부분 강한 양의 상관관계를 보임
- 독립변수와 종속변수도 약한 양의 상관관계를 보임

![상관관계 히트맵](https://github.com/whcjfdudwkd/CrabAgePrediction/assets/70883264/32d3ed04-0e3a-4490-a603-b9e4f899d909)

#### 정규분포 분석
- 종속변수의 정규분포를 확인하기 위하여 분석 진행
- Shapiro-Wilk 검정을 통한 분석 결과 종속변수는 정규성을 따르지 않음
- Q-Q 플롯(Quantile-Quantile plot)을 활용하여 시각적 정규성 확인

![Q-Q Plot 정규성 확인](https://github.com/whcjfdudwkd/CrabAgePrediction/assets/70883264/17688a19-6db0-4568-bc72-aecddaa6db04)

#### 인과관계 분석
- 종속변수가 정규분포를 따르지 않아 일반화 선형 모델(GLM)을 활용하여 독립변수와 종속변수간의 인과관계 확인
- 껍질의 무게르 제외한 나머지 변수의 p-value는 0으로 확인
- 껍질의 무게는 p-value가 0.533으로 0.05보다 큼으로 학습시 해당 데이터를 제거하고 학습

#### 다중 공선성 분석
- 독립변수간의 강한 양의 상관관계르 보여 다중 공선성 분석을 진행
- VIF(Variance Inflation Factor) 계산을 통하여 분석 진행
- VIF 값이 100이 넘는 값의 변수 제거
- VIF 결과는 아래의 표와 같음

|Variable|VIF|
|:---:|:---:|
|Length|681.495360|
|Diameter|733.627430|
|Height|40.546848|
|Weight|139.184195|
|Shucked Weight|62.125778|
|Viscera Weight|56.013000|

※일반적으로 VIF값이 10이상인 경우 다중공선성이 있다고 보고 제거하지만 그럴경우 대부분의 변수가 제거되어 100이상인 값만 제거

#### 선형 회귀 모델을 활용한 결정계수 계산
- 종속변수에 대한 독립변수의 설명하는지 보기위한 결정계수 분석 진행
- 선형회귀모델을 통해 결정계수 계산
- 결정계수 (R²): 0.3543956694513811
- 0.2가 넘어 독립변수가 유용하다고 판단
  
#### 변수선택 알고리즘을 통한 독립변수 비교
- 앞서 설명한 방법과 변수 선택 알고리즘을 통한 결과값 비교
- 변수선택 알고리즘은 SelectKBest모듈을 사용
- 결과값은 아래의 표와 같음

|Variable|VIF|
|:---:|:---:|
|Length|True|
|Diameter|True|
|Height|True|
|Weight|True|
|Shucked|False|
|Viscera|False|
|Shell|True|

## 🌏 RNN 모델 생성 및 예측
#### RNN모델 생성
- RNN모델은 TensorFlow를 사용
- 분석때와 마찬가지로 학습과 평가 데이터 분리 (8:2)
- Random_state를 분석때와 같이 하여 분석때와 같은 데이터로 분리
- 모델의 구성은 아래의 사진과 같음

![모델구성](https://github.com/whcjfdudwkd/CrabAgePrediction/assets/70883264/8b35031b-31e5-4919-8360-545d2b400ad3) 

#### 모델 학습
 - 모델은 학습데이터, 데이터 정규화, 콜백사용여부, epoch를 다르게 주어 학습
 - 각각의 내용은 아래와 같음
   1. 학습컬럼
     <br>· 분석컬럼 : 분석 후 선택한 독립변수(Height, Shucked Weight, Viscera Weight)
     <br>· 변수선택 알고리즘 : 변수선택 알고리즘에 의해 선택한 독립 변수(Length, Diameter, Height, Weight, Shell Weight)
   2. 데이터
     <br>· raw : 정규화를 안거친 데이터
     <br>· min-max : min-max 스케일링을 이용하여 데이터를 정규화한 데이터
   3. 콜백 여부
     <br>· 콜백 함수를 사용하여 val_loss가 5epoch 안에 감소하지 않으면 학습을 중단함
   4. epoch
     <br>· 모델 학습에 사용한 epoch

#### 평가
 - 모델의 평가 결과는 아래의 표와 같다

|학습컬럼|데이터|콜백 사용 여부|epoch|MSE|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|
|분석 컬럼|raw|X|100|5.906||
|분석 컬럼|raw|X|1000|6.046||
|분석 컬럼|raw|O|100|6.11|25번째|
|분석 컬럼|raw|O|1000|6.02|39번째스톱|
|분석 컬럼|min-max|X|100|6.019||
|분석 컬럼|min-max|X|1000|5.931||
|분석 컬럼|min-max|O|100|5.98|45번째 스톱|
|분석 컬럼|min-max|O|1000|5.984|44번째 스톱|
|변수선택 알고리즘|raw|X|100|5.073||
|변수선택 알고리즘|raw|X|1000|5.035||
|변수선택 알고리즘|raw|O|100|5.21|28번째 스톱|
|변수선택 알고리즘|raw|O|1000|5.22|21번째 스톱|
|변수선택 알고리즘|min-max|X|100|5.335||
|변수선택 알고리즘|min-max|X|1000|4.957||
|변수선택 알고리즘|min-max|O|100|5.47|51번째 스톱|
|변수선택 알고리즘|min-max|O|1000|5.49|45번째 스톱|

 - 직접 분석하여 선택한 독립변수보다 변수선택 알고리즘을 이용하여 선택한 독립변수의 MSE가 훨씬 낮았다(변수 선택 알고리즘 짱!)
 - EarlyStopping 콜백 정의한것 보다 안한게 대체로 결과가 좋았다(데이터가 적고 patience를 5로 설정)
 - EarlyStopping은 데이터가 많을경우 사용하여야 하고 지금처럼 적은 경우 오히려 학습이 안됌
 - 가장 좋은 결과는 변수선택알고리즘을 사용하여 독립번수를 선택하고 min-max스케일링으로 표준화한다음 1000 epoch 이상의 학습을 시키는것이 가장 좋다

## ♻️ 추후사항
 - 모델의 구조를 변경후 예측
    -> LSTM만을 이용하여 예측보다 LSTM, GRU, DENSE 레이어등을 추가하여 예측하여 MSE를 더 낮출 필요가 존재

   
