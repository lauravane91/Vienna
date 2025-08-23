# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:10:55 2024

@author: David Araque
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carga tu base de datos (ajusta el nombre del archivo según corresponda)
df = pd.read_csv('parametros_suelo_drained.csv')

# Preprocesar los datos
X = df.drop(columns=['Slope stability'])  # Características
y = df['Slope stability']  # Variable objetivo

# Escalamos las características (necesario para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear la arquitectura de la red neuronal
model = Sequential()

# Capa de entrada y primera capa oculta
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))  # 64 neuronas, función de activación ReLU

# Segunda capa oculta
model.add(Dense(units=32, activation='relu'))  # 32 neuronas

# Capa de salida (regresión, activación lineal)
model.add(Dense(units=1, activation='linear'))  # Activación linear para regresión

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluar el modelo
results = model.evaluate(X_test, y_test, verbose=0)
print(f"Resultados en el conjunto de prueba: {dict(zip(model.metrics_names, results))}")

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Graficar el rendimiento del modelo
plt.figure(figsize=(10, 6))

# Gráfico de predicciones vs valores reales
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Real value")
plt.ylabel("Prediction")
plt.title("Relationship between actual and predicted values")
plt.savefig("pred_vs_real_nn.png", dpi=300)

# Guardar pesos del modelo
model_json = model.to_json()
with open("nn_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("nn_model.weights.h5")




# Guardar scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')

