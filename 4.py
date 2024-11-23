import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Generación de datos de ejemplo
np.random.seed(42)
data = {
    "Product": (
        ["Accesorios"] * 100 + ["Audio"] * 100 + ["Electrónicos"] * 100
    ),
    "Cost price": (
        np.random.normal(5, 2, 100).tolist()
        + np.random.normal(120, 30, 100).tolist()
        + np.random.normal(130, 40, 100).tolist()
    )
}

# Creación del DataFrame
df = pd.DataFrame(data)

# Prueba ANOVA
anova_result = stats.f_oneway(
    df[df["Product"] == "Accesorios"]["Cost price"],
    df[df["Product"] == "Audio"]["Cost price"],
    df[df["Product"] == "Electrónicos"]["Cost price"],
)
print("\n--- Resultados ANOVA ---")
print(f"F-Value: {anova_result.statistic:.4f}, P-Value: {anova_result.pvalue:.4e}")

# Visualización: Boxplot para comparar distribuciones
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x="Product", y="Cost price", palette="coolwarm")
plt.title("Distribución de precios por categoría", fontsize=14, weight='bold')
plt.xlabel("Categoría de producto", fontsize=12)
plt.ylabel("Precio de costo", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
