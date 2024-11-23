import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------
# 1. Generación de Datos de Ejemplo
# -----------------------------------------

np.random.seed(42)
size = 300

# Variables independientes y dependientes
X = np.random.rand(size, 1) * 10  # Datos de entrada
y = 2.5 * X + np.random.randn(size, 1) * 5  # Salida con ruido añadido

# Crear DataFrame para los datos
data = pd.DataFrame(data=np.hstack([X, y]), columns=['X', 'y'])

# -----------------------------------------
# 2. Visualización Inicial de los Datos
# -----------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(data['X'], data['y'], color='blue', alpha=0.7)
plt.title('Distribución de Datos: X vs y', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid(True)
plt.show()

# -----------------------------------------
# 3. División en Conjunto de Entrenamiento y Prueba
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['y'], test_size=0.2, random_state=42)

# -----------------------------------------
# 4. Creación y Entrenamiento del Modelo Lineal
# -----------------------------------------

# Instanciamos el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones con el modelo entrenado
y_pred = model.predict(X_test)

# -----------------------------------------
# 5. Visualización de la Regresión Lineal
# -----------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.7, label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Modelo lineal (predicciones)')
plt.title('Regresión Lineal: Datos vs Predicciones', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------
# 6. Evaluación del Modelo
# -----------------------------------------

# Error Cuadrático Medio (MSE) y R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse:.3f}")
print(f"R² (Coeficiente de Determinación): {r2:.3f}")

# -----------------------------------------
# 7. Validación Cruzada
# -----------------------------------------

# Realizamos validación cruzada con 5 pliegues
cv_scores = cross_val_score(model, data[['X']], data['y'], cv=5, scoring='neg_mean_squared_error')
print(f"\nValidación cruzada - MSE promedio: {-cv_scores.mean():.3f}")

# -----------------------------------------
# 8. Análisis de Residuos
# -----------------------------------------

# Cálculo de residuos
residuals = y_test - y_pred

# Gráfico de residuos
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Análisis de Residuos', fontsize=16)
plt.xlabel('Predicciones', fontsize=14)
plt.ylabel('Residuos', fontsize=14)
plt.grid(True)
plt.show()

# -----------------------------------------
# 9. Análisis de Correlación
# -----------------------------------------

# Calculamos la matriz de correlación
correlation = data.corr()
print("\nCorrelación entre las variables:")
print(correlation)

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlación', fontsize=16)
plt.show()
