import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import streamlit as st

df = pd.read_parquet('datos/data_frame_final_limpio.parquet')

df_num_vac = df.select_dtypes(include=['float64', 'int32', 'int64'])
df_numerico =  df_num_vac.drop(columns=["User Score", "User Ratings Count"])
y = df_numerico["Metascore"]
X = df_numerico.drop(columns="Metascore")
pca = PCA(n_components=5)

X_pca = pca.fit_transform(X)

loaded_model = pickle.load(open('datos/modelo_recomendacion2.sav', 'rb'))
#result = loaded_model.score(X_test, Y_test)

def obtener_similitud(title):
    index = df[df['Title'].str.contains(title, case=False, na=False, regex=False)].index
    if len(index) != 0:
        nueva_fila = X_pca[index[0]].reshape(1, -1)
        distancias, indices = loaded_model.kneighbors(nueva_fila)
        # Los índices representan las recomendaciones cercanas en función de las características codificadas
        recomendaciones_cercanas = df.iloc[indices[0]]
        return recomendaciones_cercanas 
    else : 
        st.warning('El titulo no fue encontrado en la base de datos')


def obtener_similitud2(titulos):
    indices = []
    distances = []
    for titulo in titulos:
        index = df[df['Title'].str.contains(titulo, case=False, na=False, regex=False)].index
        if len(index) != 0:
            nueva_fila = X_pca[index[0]].reshape(1, -1)
            _, indice = loaded_model.kneighbors(nueva_fila)
            indices.extend(indice[0])
            distances.extend(_[0] * 1000)
        else:
            st.warning(f'El título "{titulo}" no fue encontrado en la base de datos')
    
    # Elimina duplicados y obtén las recomendaciones finales
    indices_unicos = list(set(indices))
    recomendaciones_cercanas = df.iloc[indices_unicos]
    recomendaciones_cercanas['Similitud'] = 1 / (1 + np.array(distances))
    recomendaciones_cercanas = recomendaciones_cercanas.sort_values(by='Similitud', ascending=False)

    return recomendaciones_cercanas, distances


def obtener_distance(title):
    index = df[df['Title'].str.contains(title, case=False, na=False, regex=False)].index
    if len(index) != 0:
        nueva_fila = X_pca[index[0]].reshape(1, -1)
        distancias, indices = loaded_model.kneighbors(nueva_fila)
        return distancias