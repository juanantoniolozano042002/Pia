import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV limpio
df = pd.read_csv('cleaned_sales_data.csv')

# Configuración general de estilo
sns.set(style="whitegrid")

# 1. Gráfico de Área - Tendencia de precios (Línea más visible y suavizada)
df['Order Date'] = pd.to_datetime(df['Order Date'])
price_trend = df.sort_values('Order Date')

# Crear una figura más grande
plt.figure(figsize=(12, 6))  
plt.plot(price_trend['Order Date'], price_trend['Cost price'], color='mediumblue', lw=2)  # Línea más gruesa
plt.fill_between(price_trend['Order Date'], price_trend['Cost price'], color='skyblue', alpha=0.3)  # Área debajo de la curva
plt.title('Tendencia de Precios a lo largo del Tiempo', fontsize=18, fontweight='bold')
plt.xlabel('Fecha de Pedido', fontsize=14)
plt.ylabel('Precio de Costo', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)  # Añadir cuadrícula
plt.tight_layout()
plt.savefig('tendencia_de_precios_mejorado.png')  # Guardar la imagen
plt.show()

# 2. Mejorado Boxplot - Variación de Precios (con estadísticas y mejor visualización)
plt.figure(figsize=(12, 6))  # Tamaño ajustado
sns.boxplot(y='Cost price', data=df, color='lightseagreen', width=0.6)  # Color más suave y ancho de caja ajustado
plt.title('Variación de Precios (Cost price)', fontsize=18, fontweight='bold')

# Añadir detalles sobre las estadísticas
plt.xlabel('Precios de Costo', fontsize=14)
plt.ylabel('Precio de Costo', fontsize=14)

# Añadir líneas de mediana y percentiles para mayor claridad
plt.axhline(df['Cost price'].median(), color='orange', linestyle='--', label=f'Mediana: {df["Cost price"].median():.2f}')
plt.axhline(df['Cost price'].quantile(0.25), color='yellow', linestyle=':', label=f'Q1: {df["Cost price"].quantile(0.25):.2f}')
plt.axhline(df['Cost price'].quantile(0.75), color='yellow', linestyle=':', label=f'Q3: {df["Cost price"].quantile(0.75):.2f}')
plt.legend(loc='upper right', fontsize=12)

# Cuadrícula en el eje Y
plt.grid(axis='y', linestyle='--', alpha=0.6)  # Cuadrícula más sutil
plt.tight_layout()

# Guardar y mostrar
plt.savefig('variacion_de_precios_mejorado_v2.png')  # Guardar la imagen con nuevo nombre
plt.show()


# 3. Histograma - Distribución de precios
plt.figure(figsize=(12, 6))
sns.histplot(df['Cost price'], bins=20, kde=True, color='green', alpha=0.6)
plt.title('Distribución de Precios (Cost price)', fontsize=18, fontweight='bold')
plt.xlabel('Precio de Costo', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('histograma_de_precios.png')  # Guardar la imagen
plt.show()

# 4. Gráfico de Dispersión - Relación entre el ID del Pedido y el Precio de Costo
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Order ID', y='Cost price', data=df, color='purple', s=60, alpha=0.7)
plt.title('Relación entre ID del Pedido y Precio de Costo', fontsize=18, fontweight='bold')
plt.xlabel('ID del Pedido', fontsize=14)
plt.ylabel('Precio de Costo', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('relacion_id_pedido_precio.png')  # Guardar la imagen
plt.show()

# 5. Gráfico de Pie - Proporción de Productos (Top 5 productos más vendidos)
top_products = df['Product'].value_counts().head(5)
plt.figure(figsize=(8, 8))
top_products.plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set2", n_colors=5), startangle=90, legend=False)
plt.title('Proporción de los 5 Productos Más Vendidos', fontsize=18, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig('proporcion_productos_mas_vendidos.png')  # Guardar la imagen
plt.show()
