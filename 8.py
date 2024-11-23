import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Cargar el archivo CSV
# Reemplaza con la ruta correcta de tu archivo
data = pd.read_csv('cleaned_sales_data.csv')

# Asegurarse de que la columna de fechas esté en formato datetime
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Ordenar los datos por la fecha
data = data.sort_values('Order Date')

# Crear una nueva columna 'Days Since Start' que representa el número de días desde la primera fecha
data['Days Since Start'] = (data['Order Date'] - data['Order Date'].min()).dt.days

# Seleccionar las columnas de características (X) y la variable objetivo (y)
X = data[['Days Since Start']]  # Característica: días desde el inicio
y = data['Cost price']          # Variable objetivo: precio del producto

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar las predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) para evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(data['Order Date'], data['Cost price'], color='blue', label='Datos reales')  # Datos reales
plt.plot(data['Order Date'], model.predict(X), color='red', label='Modelo de regresión')  # Línea de regresión
plt.title('Predicción de precios con regresión lineal')
plt.xlabel('Fecha')
plt.ylabel('Precio del Producto')
plt.legend()
plt.xticks(rotation=45)  # Rotar las fechas para mejor visibilidad
plt.tight_layout()  # Ajustar el layout para que todo encaje bien
plt.show()

# Predicciones futuras
# Crear un rango de fechas para predecir los próximos 30 días
future_days = pd.date_range(start=data['Order Date'].max(), periods=31, freq='D')
future_days_int = (future_days - data['Order Date'].min()).days  # Convertir fechas a días

# Convertir future_days_int a un array de NumPy para poder usar reshape
future_days_int = np.array(future_days_int)

# Predecir los precios para los días futuros
future_prices = model.predict(future_days_int.reshape(-1, 1))

# Mostrar las predicciones para los próximos 30 días
predictions = pd.DataFrame({'Date': future_days, 'Predicted Price': future_prices})
print("\nPredicciones para los próximos 30 días:")
print(predictions)
