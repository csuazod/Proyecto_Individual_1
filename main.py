import pandas as pd 
import numpy as np
import sklearn
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.neighbors import NearestNeighbors

app = FastAPI(title='PROYECTO INDIVIDUAL 01 - Machine Learning Operations (MLOps)',
            description='API de datos y recomendaciones de películas basado en machine learning')

df = pd.read_csv('API_movies_dataset.csv')
df_movies = pd.read_csv('ML_movies.csv')

@app.get('/')
async def index():
    return {'Bienvenido a la API de recomedación y consulta de peliculas. Por favor dirigite a /docs'}


@app.get('/peliculas_idioma/{idioma}')
# Funcion de consulta del numero de peliculas por idioma
def peliculas_idioma(idioma):
    # Selecciona todas las películas del DataFrame 'df' cuya columna 'original_language' contiene el idioma especificado.
    # La función lambda se utiliza para aplicar la operación a cada valor de la columna 'original_language'.
    peliculas_idioma = df[df['original_language'].apply(lambda x: idioma in str(x) if pd.notnull(x) else False)]
    
    # Elimina las filas duplicadas de la película para evitar contar varias veces una misma película.
    peliculas_idioma = peliculas_idioma.drop_duplicates(subset='id')
    
    # Cuenta la cantidad de películas restantes después de eliminar los duplicados
    respuesta = len(peliculas_idioma)
    
    # Devuelve los resultados en un diccionario con claves legibles
    return {'idioma': idioma, 'cantidad de peliculas': respuesta}


@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula):
    
    info_pelicula = df[(df['title'] == pelicula)].drop_duplicates(subset='title')
    pelicula_nombre = info_pelicula['title'].iloc[0]
    duracion_pelicula = str(info_pelicula['runtime'].iloc[0])
    año_pelicula = str(info_pelicula['release_year'].iloc[0])

    return {'pelicula':pelicula_nombre, 'duracion min':duracion_pelicula, 'año':año_pelicula}


@app.get('/franquicia/{franquicia}')
# Funcion de consulta del numero de colecciones de peliculas, su ganancia total y ganancia promedio 
def franquicia(franquicia):

    lista_pelis_franquicia = df[(df['name_btc'] == franquicia)].drop_duplicates(subset='id')

    cantidad_pelis_franq = (lista_pelis_franquicia).shape[0]
    revenue_franq = lista_pelis_franquicia['revenue'].sum()
    promedio_franq = revenue_franq/cantidad_pelis_franq

    return {'franquicia':franquicia, 'cantidad':cantidad_pelis_franq, 'ganancia_total':revenue_franq, 'ganancia_promedio': promedio_franq}

@app.get('/peliculas_pais/{pais}')
# Funcion de consulta del numero de peliculas por pais 2
def peliculas_pais(pais):
    # Selecciona todas las películas del DataFrame 'df' cuya columna 'production_countries' contiene el país especificado.
    # La función lambda se utiliza para aplicar la operación a cada valor de la columna 'production_countries'.
    peliculas_pais = df[df['production_countries'].apply(lambda x: pais in str(x) if pd.notnull(x) else False)]
    
    # Elimina las filas duplicadas de la película para evitar contar varias veces una misma película.
    peliculas_pais = peliculas_pais.drop_duplicates(subset='id')
    
    # Cuenta la cantidad de películas restantes después de eliminar los duplicados.
    respuesta = len(peliculas_pais)
    
    # Devuelve los resultados en un diccionario con claves legibles.
    return {'pais': pais, 'cantidad de peliculas': respuesta}


@app.get('/productoras_exitosas/{productora}')
# Funcion de consulta del numero de peliculas por productora, ganancias totales y numero de peliculas
def productoras_exitosas(productora):

    lista_pelis_productoras = df[(df['production_companies'] == productora)].drop_duplicates(subset='id')

    cantidad_pelis_prod = (lista_pelis_productoras).shape[0]
    revenue_prod = lista_pelis_productoras['revenue'].sum()


    return {'productora':productora, 'ganancia_total':revenue_prod, 'cantidad':cantidad_pelis_prod}


# Funcion Machine Learning - "Modelo de K Vecinos mas Cercanos"

def movie_recommendation(movie_title):

    movie_title = movie_title.lower()
    # Buscar la película por título en la columna 'title'
    movie = df_movies[df_movies['title'].str.lower() == movie_title]

    if len(movie) == 0:
        return "La película no se encuentra en la base de datos."

    # Obtener el género y la popularidad de la película
    movie_genre = movie['genre_names'].values[0]
    movie_popularity = movie['popularity'].values[0]

    # Crear una matriz de características para el modelo de vecinos más cercanos
    features = df_movies[['popularity']]
    genres = df_movies['genre_names'].str.get_dummies(sep=' ')
    features = pd.concat([features, genres], axis=1)

    # Manejar valores faltantes (NaN) reemplazándolos por ceros
    features = features.fillna(0)

    # Crear el modelo de vecinos más cercanos
    nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn_model.fit(features)

    # Encontrar las películas más similares
    _, indices = nn_model.kneighbors([[movie_popularity] + [0] * len(genres.columns)], n_neighbors=6)

    # Obtener los títulos de las películas recomendadas
    recommendations = df_movies.iloc[indices[0][1:]]['title']

    return recommendations

@app.get("/recomendacion/{movie_title}", tags=['Machine Learning'])
def movie_recommendation(movie_title: str):
    
    recommended_movies = movie_recommendation(movie_title)
    return {"recommended_movies": recommended_movies.tolist()}
