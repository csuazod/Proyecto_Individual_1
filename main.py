import pandas as pd 
import numpy as np
import sklearn
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.neighbors import NearestNeighbors
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title='PROYECTO INDIVIDUAL 01 - Machine Learning Operations (MLOps)',
            description='API de datos y recomendaciones de películas basado en machine learning')

df = pd.read_csv('API_movies_dataset.csv')

data= pd.read_csv('Datasets/DEset.csv')
df_Dset = pd.DataFrame(data)
df_Dset.drop_duplicates(subset='id',inplace=True)

credits=pd.read_csv('Datasets/credits_filtered.csv')
df_credits=pd.DataFrame(credits)

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

ml_data_preliminar= pd.merge(df_Dset, df_credits, on='id')
ml_data=ml_data_preliminar[ml_data_preliminar['vote_count'] > 250]

ml_data['cast_filtered'] = ml_data['cast_filtered'].str.replace("[\[\]',]", "").str.strip()
ml_data['genres_filtered'] = ml_data['genres_filtered'].str.replace("[\[\]',]", "").str.strip()
ml_data['production_companies_filtered'] = ml_data['production_companies_filtered'].str.replace("[\[\]',]", "").str.strip()

selected_features = ['genres_filtered','tagline','cast_filtered','crew_filtered','overview','production_companies_filtered']

for feature in selected_features:
  ml_data[feature] = ml_data[feature].fillna('')

combined_features = (ml_data['genres_filtered']+ ' ').str.repeat(25)+ (ml_data['tagline'] + ' ').str.repeat(10) + (ml_data['cast_filtered'] + ' ').str.repeat(20) + (ml_data['crew_filtered']+' ').str.repeat(15)+(ml_data['production_companies_filtered']+' ').str.repeat(20)+(ml_data['overview']).str.repeat(10)

vectorizer = TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
movies_list= ml_data['title'].tolist()

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):

    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    #Interaction with user
    movie_name=titulo
    #Closest match
    find_close_match = difflib.get_close_matches(movie_name, movies_list)

    #Closest match possible in the data
    close_match = find_close_match[0]
    id_of_the_movie = ml_data[ml_data.title == close_match]['id'].values[0]

    #Obtain the more similar movie
    similarity_score = list(enumerate(similarity[id_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

    i = 1
    recommendation_movies = []
    added_movies = set()  # Utilizar un conjunto para rastrear las películas agregadas

    for movie in sorted_similar_movies:
        id = movie[0]
        filtered_df = ml_data[ml_data['id'] == id]
        if not filtered_df.empty:
            title_from_id = filtered_df['title'].values[0]
        if title_from_id not in added_movies:  # Verificar si la película ya está en la lista
            recommendation_movies.append(title_from_id)
            added_movies.add(title_from_id)  # Agregar la película al conjunto de películas agregadas
            i += 1
        if i >= 6:
            break  # Detener el bucle si se han agregado suficientes películas a la lista
    
    return {'lista recomendada': recommendation_movies}