import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv('cleaned_sales_data.csv')

# Verificar las columnas del DataFrame
print(df.columns)

# Suponiendo que la columna que contiene las descripciones de los productos es 'Product', ajusta según tu archivo
# Si la columna contiene valores faltantes (NaN), los eliminamos
text_data = ' '.join(df['Product'].dropna())  # Cambia 'Product' por el nombre correcto si es necesario

# Crear la nube de palabras
stopwords = set(STOPWORDS)  # Definir las palabras a excluir

wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color='white').generate(text_data)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Quitar los ejes
plt.show()

