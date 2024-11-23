import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Cargar el archivo CSV limpio
df = pd.read_csv('cleaned_sales_data.csv')

# Limpiar valores nulos
df.dropna(inplace=True)

# Estadísticas descriptivas generales
print("Estadísticas descriptivas generales:")
print(df.describe())

# Identificar las entidades y relaciones
# Asumimos las entidades: Fecha, Pedido (Order ID), Producto y Precio
entities = ['Order Date', 'Order ID', 'Product', 'Cost price']
relations = [
    ('Order Date', 'Order ID'),
    ('Order ID', 'Product'),
    ('Product', 'Cost price')
]

# Crear un diagrama de relaciones
G = nx.DiGraph()
G.add_nodes_from(entities)
G.add_edges_from(relations)

# Dibujar el diagrama
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Diagrama de Entidades y Relaciones")
plt.show()

# Agrupación de datos por producto y estadísticas
grouped = df.groupby('Product').agg(
    cantidad=('Product', 'count'),
    costo_promedio=('Cost price', 'mean'),
    costo_total=('Cost price', 'sum')
)
print("\nEstadísticas agrupadas por producto:")
print(grouped)

# Guardar las estadísticas agrupadas en un archivo CSV
grouped.to_csv('grouped_statistics.csv')
