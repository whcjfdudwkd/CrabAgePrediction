import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from statsmodels.genmod import families
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import probplot
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

# 모든 경고 메시지를 무시하고 출력하지 않음
warnings.filterwarnings("ignore")

data = pd.read_csv('./data/CrabAgePrediction.csv')
data = data.reset_index(drop=1)

data.loc[data['Sex'] == 'M', 'sex'] = 0
data.loc[data['Sex'] == 'F', 'sex'] = 1
data.loc[data['Sex'] == 'I', 'sex'] = 2

# data = data.sort_values('Age', ascending=False)

# 예시 데이터 (X는 특성, y는 타겟)
X = data[['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
y = data[['Age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 상관관계 계산
test = data[['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Age']]
correlation_matrix = test.corr()

# 결과 출력
save_path = './image/'
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('상관관계 히트맵')
plt.tight_layout()
# plt.show()
plt.savefig(save_path+'상관관계 히트맵')

""" 히스토그램 그리기 """
# Sturges' Formula
# num_bins = 1 + int(np.log2(len(data)))

# Square Root Choice
# num_bins = int(np.sqrt(len(data)))

# Scott's Normal Reference Rule
# num_bins = int(3.5 * np.std(data['Age']) / (len(data) ** (1/3)))
#
# Freedman-Diaconis' Rule
# iqr = np.percentile(data['Age'], 75) - np.percentile(data['Age'], 25)
# num_bins = int(2 * iqr / (len(data) ** (1/3)))
# plt.clf()
# plt.hist(data['Age'], bins=num_bins, density=True, alpha=0.7, color='blue')
# plt.title('데이터 분포')
# plt.xlabel('값')
# plt.ylabel('빈도')
# plt.show()

# 종속변수의 정규분포 확인

stat, p_value = shapiro(data['Age'])
alpha = 0.05
if p_value > alpha:
    print("데이터는 정규 분포를 따릅니다.")
else:
    print("데이터는 정규 분포를 따르지 않습니다.")

plt.clf()
y_train_flattened = data['Age'].to_list()
probplot(y_train_flattened, dist='norm', plot=plt)
plt.title('Q-Q Plot')
# plt.show()
plt.savefig(save_path+'Q-Q Plot 정규성 확인')


# Poisson 분포를 사용하여 일반화 선형 모델 적합
# P-value 확인
X = data[['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
y = data[['Age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = sm.GLM(y_train, X_train, family=families.Poisson()).fit()
print(model.summary())

# 다중 공선성 분석
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight']]
y = data[['Age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X에는 독립 변수들이 들어가야 함
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

from sklearn.decomposition import PCA
# X에는 독립 변수들이 들어가야 함
X = data[['Height', 'Shucked Weight', 'Viscera Weight']]
y = data[['Age']]
pca = PCA()
X_pca = pca.fit_transform(X)
# PCA 결과 확인
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# 선형 회귀 모델 생성 및 훈련
X = data[['Height', 'Shucked Weight', 'Viscera Weight']]
y = data[['Age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# 테스트 세트에 대한 예측
y_pred = model.predict(X_test)
# 결정계수 계산
r2 = r2_score(y_test, y_pred)
print(f"결정계수 (R²): {r2}")


# SelectKBest모듈을 활용한 변수 선택 방법
X = data[['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
y = data[['Age']]

kbest = SelectKBest(score_func=f_regression, k=5)
fit = kbest.fit(X, y)
selected_features = fit.get_support()
print(fit.feature_names_in_)
print(selected_features)

