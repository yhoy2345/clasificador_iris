import pandas as pd

# Ruta al archivo iris.data extraído
dataset_path = r'C:\Users\ESTUDIANTE\Documents\STAR\iris\iris.data'  # Ruta correcta

# Cargar el dataset
df = pd.read_csv(dataset_path, header=None)

# Definir los nombres de las columnas
df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']

# Mostrar las primeras filas del dataset
print(df.head())
