import pandas as pd

# Cargar el archivo CSV original
df = pd.read_csv('sales_data.csv')

# Seleccionar solo las columnas necesarias
columns_needed = ['Order Date', 'Order ID', 'Product', 'Cost price']
df_cleaned = df[columns_needed]

# Eliminar las filas que contienen valores nulos
df_cleaned = df_cleaned.dropna()

# Guardar el DataFrame limpio en un nuevo archivo CSV
df_cleaned.to_csv('cleaned_sales_data.csv', index=False)

# Mostrar los primeros registros para verificar
print(df_cleaned.head())
