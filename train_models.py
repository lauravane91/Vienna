# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1️⃣ Cargar la base de datos
df = pd.read_csv('parametros_suelo_drained.csv')

# 2️⃣ Separar características y variable objetivo
X = df.drop(columns=['Slope stability'])
y = df['Slope stability']

# 3️⃣ Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ================= Random Forest =================
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
print(f"R²: {r2_score(y_test, y_pred_rf):.4f}")

# Guardar modelo RF
joblib.dump(rf_model, 'rf_model.pkl')

# ================= SVM =================
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\nSVM:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svm):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_svm):.4f}")
print(f"R²: {r2_score(y_test, y_pred_svm):.4f}")

# Guardar modelo SVM
joblib.dump(svm_model, 'svm_model.pkl')

# ================= Red Neuronal =================
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='linear'))
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

y_pred_nn = nn_model.predict(X_test)

print("\nRed Neuronal:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_nn):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_nn):.4f}")
print(f"R²: {r2_score(y_test, y_pred_nn):.4f}")

# Guardar arquitectura y pesos NN
model_json = nn_model.to_json()
with open("nn_model.json", "w") as json_file:
    json_file.write(model_json)
nn_model.save_weights("nn_model.weights.h5")

# ================= Guardar scaler =================
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Entrenamiento completado. Todos los modelos y el scaler se han guardado correctamente.")
