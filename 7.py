import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar el archivo CSV
data = pd.read_csv('cleaned_sales_data.csv')

# Mostrar los primeros registros del archivo
print(data.head())

# Si es posible, se pueden usar más columnas para el clustering. Aquí solo usamos 'Cost price'.
X = data[['Cost price']]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir el número de clusters
n_clusters = 3  # Puedes probar con 2, 3, 4, 5, etc.

# Crear el modelo KMeans y ajustarlo
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Asignar las etiquetas de los clusters al DataFrame original
data['Cluster'] = kmeans.labels_

# Ver los resultados
print(data.head())

# Graficar los resultados (en este caso, solo un feature, 'Cost price')
plt.figure(figsize=(8, 6))
plt.scatter(data['Cost price'], [0]*len(data), c=data['Cluster'], cmap='viridis', s=100, edgecolor='k', alpha=0.7)
plt.title('Clusters de productos según el precio')
plt.xlabel('Precio del Producto')
plt.yticks([])  # No mostrar eje Y, ya que es innecesario para este gráfico
plt.colorbar(label='Cluster')
plt.show()

# Ver el número de elementos en cada cluster
print(data.groupby('Cluster').size())

# Opcional: También puedes experimentar con más características si están disponibles
# Por ejemplo, si hay columnas como 'Product Category', 'Order Date', etc., puedes agregarlas al clustering.
# X = data[['Cost price', 'Other column(s)']]  # Añadir más columnas según sea necesario

