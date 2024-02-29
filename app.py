import streamlit as st
import pandas as pd
import knn

# Cargar datos y modelo
df = pd.read_parquet('datos/data_frame_final_limpio.parquet')

st.title("Sistema de Recomendación de Videojuegos")

# Elementos en la barra lateral
st.sidebar.subheader('Opciones')
titulos_ingresados = st.sidebar.multiselect("Ingrese títulos de juegos:", df['Title'].tolist())

# Botón para obtener recomendaciones
if st.sidebar.button("Obtener Recomendaciones"):
    # Verificar si se ingresaron títulos
    if not titulos_ingresados:
        st.warning("Por favor, ingrese al menos un título de juego.")
    else:
        # Obtener recomendaciones
        recomendaciones, _ = knn.obtener_similitud2(titulos_ingresados)
        
        # Mostrar resultados en el cuerpo principal de la página
        st.subheader("Recomendaciones Cercanas:")
        st.table(recomendaciones)
        st.table(_)