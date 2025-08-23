
import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['Random Forest', 'SVM', 'Neural Networks']

# Metrics (replace with your actual values if needed)
mae = [0.15, 0.1959, 0.1000]
rmse = [0.23, 0.36, 0.16]
r2 = [0.8936, 0.7532, 0.9518]

# Bar positions
x = np.arange(len(models))
width = 0.25

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Add bars for each metric
bars1 = ax.bar(x - width, mae, width, label='MAE')
bars2 = ax.bar(x, rmse, width, label='RMSE')
bars3 = ax.bar(x + width, r2, width, label='R² Score')

# Labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison of Machine Learning Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the figure
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()


