import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('./data/CrabAgePrediction.csv')

# 분석
# X = data[['Height', 'Shucked Weight', 'Viscera Weight']]
# y = data[['Age']]

# 변수선택알고리즘
X = data[['Length', 'Diameter', 'Height', 'Weight', 'Shell Weight']]
y = data[['Age']]

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()
# X 데이터를 스케일링
X_scaled = scaler.fit_transform(X)
# 스케일링된 데이터로 새로운 데이터프레임 생성
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(None, 5)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
    tf.keras.layers.Dense(1, activation='linear')  # 0~30의 값을 표출하도록 수정
])

# EarlyStopping 콜백 정의
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, epochs=1000, batch_size=32,
          validation_data=(X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])), y_test))

# model.fit(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, epochs=1000, batch_size=32,
#           validation_data=(X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])), y_test),
#           callbacks=[early_stopping])

predictions = model.predict(X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])))

mse = mean_squared_error(y_test, predictions)
y_test = y_test.reset_index(drop=1)
print(f'Mean Squared Error: {mse}')

model.summary()