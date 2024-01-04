딥러닝을 활용한 게의 나이 예측

## 🖥️ 프로젝트 소개
Kaggle에 존재하는 게 데이터를 사용하여 게의 나이를 예측
<br>

## 🕰️ 개발 기간
* 24.01.03일 ~

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
|:------:|:--------:|
|Length|688.146252|
|Diameter|742.886099|
|Height|40.820789|
|Weight|412.753667|
|Shucked|99.669938|
|Viscera|62.339241|
|Shell|81.318543|

※일반적으로 VIF값이 10이상인 경우 다중공선성이 있다고 보고 제거하지만 변수가 적어 대부분의 변수가 제거되어 100이상인 값만 제거

#### 선형 회귀 모델을 활용한 결정계수 계산
- 종속변수에 대한 독립변수의 설명하는지 보기위한 결정계수 분석 진행
- 선형회귀모델을 통해 결정계수 계산
- 결정계수(R²) : 0.5094260894688227
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
