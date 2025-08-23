# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:30:43 2024

@author: David Araque
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv('D:/Trabajo/Aether/NZGS2025/PRUEBAS/parametros_suelo_drained.csv')

# Mostrar las primeras filas del DataFrame
print(df.head())

# Obtener el resumen estadístico y redondear a 2 decimales
resumen = df.describe().round(2)

# Crear un índice descriptivo
resumen.index = [
    'Count',    # count
    'Media',                  # mean
    'Desviación Estándar',    # std
    'Mínimo',                 # min
    '25% Cuartil',           # 25%
    'Mediana',                # 50%
    '75% Cuartil',           # 75%
    'Máximo'                  # max
]

# Reiniciar el índice para convertirlo en una columna
resumen = resumen.reset_index()

# Visualizar el resumen en una tabla
plt.figure(figsize=(16, 12))  # Aumentar el tamaño de la figura
plt.table(cellText=resumen.values, colLabels=resumen.columns, loc='center', cellLoc='center', colColours=['#f2f2f2'] * len(resumen.columns))
plt.axis('off')  # Ocultar los ejes
plt.title('Resumen Estadístico de los Datos', fontsize=16)

# Guardar la figura con alta resolución
plt.savefig('resumen_estadistico.png', dpi=300)

plt.show()


# Crear la figura con 5 subgráficos
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Crear gráficos de violín para cada variable en un subgráfico separado
sns.violinplot( y='Peso específico (kN/m³)', data=df, ax=axes[0,0])
axes[0,0].set_ylabel('$\gamma$($Kn/m^3$)')

sns.violinplot( y= 'Cohesión (kPa)', data=df, ax=axes[0,1])
axes[0,1].set_ylabel('C (kPa)')

sns.violinplot( y='Ángulo de fricción (°)', data=df, ax=axes[0,2])
axes[0,2].set_ylabel('$\phi$')

sns.violinplot( y= 'Ángulo del talud (°)', data=df, ax=axes[1,0])
axes[1,0].set_ylabel(r'$\alpha$')

sns.violinplot( y='Altura del talud (m)', data=df, ax=axes[1,1])
axes[1,1].set_ylabel('H(m)')


# Eliminar el gráfico vacío en la posición [1, 2] (última celda en la segunda fila)
axes[1, 2].axis('off')  # Esto oculta el gráfico vacío

# Añadir títulos generales

fig.tight_layout()  # Ajustar el espacio entre los subgráficos
fig.subplots_adjust(top=0.85)  # Para dar espacio al título general

# Mostrar el gráfico
plt.show()

# Crear la figura con 2 filas y 3 columnas (espacio para 6 gráficos)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Crear gráficos de boxplot y colocar las etiquetas sobre el eje Y
sns.boxplot(y='Peso específico (kN/m³)', data=df, ax=axes[0, 0])
axes[0, 0].set_ylabel(r'$\gamma$ (kN/m$^3$)')

sns.boxplot(y='Cohesión (kPa)', data=df, ax=axes[0, 1])
axes[0, 1].set_ylabel('C (kPa)')

sns.boxplot(y='Ángulo de fricción (°)', data=df, ax=axes[0, 2])
axes[0, 2].set_ylabel(r'$\phi$')

sns.boxplot(y='Ángulo del talud (°)', data=df, ax=axes[1, 0])
axes[1, 0].set_ylabel(r'$\alpha$')

sns.boxplot(y='Altura del talud (m)', data=df, ax=axes[1, 1])
axes[1, 1].set_ylabel('H (m)')

# Eliminar el gráfico vacío en la posición [1, 2] (última celda en la segunda fila)
axes[1, 2].axis('off')  # Esto oculta el gráfico vacío


# Ajustar el layout para que no se solapen
fig.tight_layout()
fig.subplots_adjust(top=0.85)  # Para dar espacio al título general

# Mostrar el gráfico
plt.show()
