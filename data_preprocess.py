import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/CrabAgePrediction.csv')
data = data.reset_index(drop=1)

data.loc[data['Sex'] == 'M', 'sex'] = 0
data.loc[data['Sex'] == 'F', 'sex'] = 1
data.loc[data['Sex'] == 'I', 'sex'] = 2

# data = data.sort_values('Age', ascending=False)

# 예시 데이터 (X는 특성, y는 타겟)
X = data[['sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
y = data[['Age']]

# train_test_split 함수를 사용하여 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 상관관계 계산
test = data[['sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Age']]
correlation_matrix = test.corr()

# 결과 출력
save_path = './image/'
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('상관관계 히트맵')
# plt.show()
plt.savefig(save_path+'상관관계 히트맵')

num_bins = 1 + int(np.log2(len(data)))
num_bins = int(np.sqrt(len(data)))
# num_bins = int(3.5 * np.std(data['Age']) / (len(data) ** (1/3)))

# iqr = np.percentile(data['Age'], 75) - np.percentile(data['Age'], 25)
# num_bins = int(2 * iqr / (len(data) ** (1/3)))

# 히스토그램 그리기
plt.clf()
plt.hist(data['Age'], bins=num_bins, density=True, alpha=0.7, color='blue')
plt.title('데이터 분포')
plt.xlabel('값')
plt.ylabel('빈도')
plt.show()