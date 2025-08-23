# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:15:51 2024

@author: David Araque
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar la base de datos
df = pd.read_csv('parametros_suelo_drained.csv')


# Dividir características (X) y etiqueta (y)
X = df.drop(columns=['Slope stability'])  # Eliminar la columna objetivo
y = df['Slope stability']  # Columna objetivo para predecir

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo Random Forest con ajustes para evitar sobreajuste
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predecir la estabilidad de taludes (Slope stability)
slope_stability_pred = rf_model.predict(X_test_scaled)

# Crear un DataFrame consolidado con los resultados
results_df = pd.DataFrame({
    "Valor Real (Slope stability)": y_test.values,
    "Predicción (Slope stability)": slope_stability_pred.round(2)
})

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, slope_stability_pred)
r2 = r2_score(y_test, slope_stability_pred)

mae = mean_absolute_error(y_test, slope_stability_pred)
mse = mean_squared_error(y_test, slope_stability_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# Validación cruzada para verificar la generalización del modelo
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"R² promedio de la validación cruzada: {cv_scores.mean():.4f}")

# Exportar los resultados a un archivo CSV
results_df.to_csv("predicciones.csv", index=False)

# Graficar la relación entre valores reales y predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, slope_stability_pred, alpha=0.7, edgecolors="k", label="Datos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2, label="Línea ideal")
plt.title(f"Relationship between actual and predicted values\nMSE: {mse:.2f}, R²: {r2:.4f}")
plt.xlabel("Real value (Slope stability)")
plt.ylabel("Prediction (Slope stability)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.savefig("pred_vs_real_rf.png", dpi=300)

# Graficar los residuos (errores) para verificar si hay sobreajuste
residuals = y_test - slope_stability_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.7, edgecolors="k")
plt.axhline(0, color='red', lw=2, linestyle='dashed')
plt.title("Gráfico de Residuos")
plt.xlabel("Valor Real (Slope stability)")
plt.ylabel("Residuos (Error)")
plt.grid(alpha=0.3)
plt.show()
plt.savefig("residuos_rf.png", dpi=300)
