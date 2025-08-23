# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:20:28 2024

@author: David Araque
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar los datos desde el archivo CSV
df = pd.read_csv('parametros_suelo_drained.csv')


# Suponiendo que la última columna es la variable objetivo
X = df.drop('Slope stability', axis=1)  # Reemplaza 'Slope stability' con el nombre de tu columna objetivo
y = df['Slope stability']  # Reemplaza con el nombre de la columna objetivo

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos (SVM RBF es sensible a las escalas)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el clasificador SVM con kernel RBF
model = SVR(kernel='rbf')

# Entrenar el modelo
model.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo con métricas de regresión
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Gráfico de dispersión entre predicciones y valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Línea de referencia
plt.title('Relationship between actual and predicted values')
plt.xlabel('Real values')
plt.ylabel('Predictions')
plt.grid(True)
plt.show()
plt.savefig("pred_vs_real_svm.png", dpi=300)

# Calcular residuos
y_residuals = y_test - y_pred

# Graficar los residuos
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_residuals)), y_residuals, color='purple', label='Residuos')
plt.axhline(0, color='red', linestyle='--', label='Cero Residual')
plt.title('Gráfico de Residuos')
plt.xlabel('Índice de las muestras')
plt.ylabel('Residuos')
plt.legend()
plt.show()


# Reducir las dimensiones a 2D para visualización (si el número de características > 2)
if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    X_test_2d = pca.transform(X_test_scaled)
else:
    X_train_2d = X_train_scaled
    X_test_2d = X_test_scaled

# Ajustar el modelo de SVM nuevamente con las características reducidas (solo para visualización)
model_2d = SVR(kernel='rbf')
model_2d.fit(X_train_2d, y_train)

# Predicciones usando las características reducidas
y_pred_2d = model_2d.predict(X_test_2d)

# Gráfico de vectores de soporte
plt.figure(figsize=(12, 8))
plt.scatter(X_train_2d[:, 0], y_train, color='blue', label='Datos de Entrenamiento')
plt.scatter(X_test_2d[:, 0], y_test, color='orange', label='Datos de Prueba')
plt.scatter(X_test_2d[:, 0], y_pred_2d, color='green', label='Predicciones')
plt.axhline(0, color='red', linestyle='--', label='Línea Residual')
plt.title("Visualización del Modelo SVM (Reducción a 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Valor de la Variable Objetivo")
plt.legend()
plt.show()

# Vectores de soporte
plt.figure(figsize=(12, 8))
plt.scatter(X_train_2d[:, 0], y_train, color='blue', label='Datos de Entrenamiento')
plt.scatter(model_2d.support_vectors_[:, 0], y_train[model_2d.support_], color='purple', edgecolor='k', 
            marker='o', s=100, label='Vectores de Soporte')
plt.title("Vectores de Soporte")
plt.xlabel("Componente Principal 1")
plt.ylabel("Valor de la Variable Objetivo")
plt.legend()
plt.show()