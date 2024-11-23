import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cargar el conjunto de datos
data = pd.read_csv('cleaned_sales_data.csv')

# Mostrar las primeras filas del conjunto de datos
print(data.head())

# Extraer características de la columna 'Order Date'
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day
data['Hour'] = data['Order Date'].dt.hour
data['Minute'] = data['Order Date'].dt.minute

# Selección de características y variable objetivo
X = data[['Year', 'Month', 'Day', 'Hour', 'Minute']]  # Puedes incluir otras variables relevantes
y = data['Cost price']

# Convertir variables categóricas como 'Product' a variables numéricas usando One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos para que todos estén en la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo con Mean Squared Error (MSE) y R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar los resultados
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Evaluar el modelo usando validación cruzada para obtener una evaluación más robusta
cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation MSE (promedio): {-cross_val_scores.mean():.2f}')
