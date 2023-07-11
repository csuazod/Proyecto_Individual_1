# PROYECTO INDIVIDUAL 01 - Machine Learning Operations

#### Desarrollado por: Cristian Suazo

# Video de Demostración

En este video, encontraran una descripción paso a paso de cómo utilizar la aplicación, segun las consultas solicitadas para el proyecto.

<a href="" target="_blank">Video de Demostración</a>

---

# Diccionario de Carpetas y Archivos En Repositorio

```python

    "Datasets": "En esta carpeta se guardan los archivos descomprimidos de los dataset que se utilizaron",

    "EDA": "Esta carpeta contiene un archivo en formato .ipynb que se utilizó para realizar el análisis
	exploratorio de los datos.",


    "Transformacion_movies_dataset.ipynb": "Es un archivo que incluyen procesos de transformación y limpieza de datos, tratamiento de
	columnas anidadas, sustitución de valores nulos, revisión de tipos de datos y códigos para eliminar
	valores repetidos o con formatos distintos a los que corresponden en su columna.",

    "API_movies_dataset.csv": "Archivo que contiene todas las funciones para poder desarrollar la API con FastAPI."

    "ML_movies.csv": "Es un archivo en formato .csv en el que se encuentran los datos necesarios para el desarrollo del sistema 
    de recomendacion ML.",

    "Transformacion_credits.ipynb": "Es un archivo en formato .ipynb donde se realiza el proceso
	ETL del archivo credits.csv.",

    Funciones_ML_movies_dataset.ipynb: "Es un archivo en formato .ipynb en el que se desarrolló la función
	para crear un modelo de aprendizaje utilizando el método 'vecinos más cercanos'.",

    "main.py": "Archivo que contiene todo el código de la API desarrollada con FastAPI.",

    "requirements.txt": "Archivo útil para realizar el despliegue en Render."

```





# ETL y EDA

## Exploración y Limpieza de Datos


Durante la exploración y limpieza de los datos iniciales, desanidé campos como `belongs_to_collection` y `production_companies`, lo cual me permitió extraer información relevante que posteriormente utilicé en consultas de la API. 

Además, para garantizar la integridad de mi conjunto de datos, rellené los valores nulos en los campos `revenue` y `budget` con el número 0. Esta acción me permitió manejar adecuadamente los datos faltantes y evitar posibles problemas en etapas posteriores del proyecto.

Eliminé los registros con valores nulos en el campo `release date` y aseguré que todas las fechas estén en el formato AAAA-mm-dd. Esta transformación fue esencial para analizar y comparar de manera efectiva las fechas de estreno de las películas.

Creé una nueva columna llamada `release_year`, la cual me permitió extraer el año de lanzamiento de cada película. Esta adición me brindó la capacidad de realizar análisis basados en años y observar las tendencias a lo largo del tiempo, brindando una perspectiva temporal valiosa.

Con el objetivo de evaluar el rendimiento financiero de las películas, calculé un nuevo campo llamado `return`, el cual representa el retorno de inversión al dividir los campos `revenue` y `budget`. En casos donde no había datos disponibles para el cálculo, asigné el valor 0. Esta métrica proporcionó información valiosa sobre el rendimiento financiero de las películas, permitiéndome tomar decisiones informadas basadas en datos.

Por último, para simplificar mi conjunto de datos y enfocarme en las variables más significativas para mi proyecto, eliminé las columnas que no eran relevantes, tales como `video`, `imdb_id`, `adult`, `original_title`, `poster_path` y `homepage`. Esta acción me permitió reducir el ruido en mis datos.


## Diccionario de datos


El diccionario de datos asociado a datasets_final.csv proporciona descripciones detalladas de cada columna en este conjunto de datos, lo cual es fundamental para comprender la información contenida en él. Al conocer el propósito y el significado de cada columna, puedo tomar decisiones más informadas y realizar análisis más precisos y relevantes.


A continuación se muestra un diccionario que describe cada columna en el conjunto de datos del archivo datasets_final.csv:

```python
column_description = {
    'id': 'ID de la película',
    'title': 'Título de la película',
    'overview': 'Descripción de la película',
    'popularity': 'Popularidad de la película',
    'vote_average': 'Promedio de votos de la película',
    'vote_count': 'Número de votos de la película',
    'status': 'Estado de la película',
    'original_language': 'Idioma original de la película',
    'runtime': 'Duración de la película en minutos',
    'budget': 'Presupuesto de la película',
    'revenue': 'Ingresos generados por la película',
    'tagline': 'Lema de la película',
    'id_btc': 'ID de la película en BTC',
    'name_btc': 'Nombre de la película en BTC',
    'iso_639_1': 'Código ISO 639-1 del idioma',
    'release_year': 'Año de lanzamiento de la película',
    'return': 'Relación entre ingresos y presupuesto de la película',
    'companies_id': 'ID de las compañías de producción',
    'companies_name': 'Nombres de las compañías de producción',
    'countries_iso': 'Códigos ISO de los países de producción',
    'countries_name': 'Nombres de los países de producción',
    'release_date': 'Fecha de lanzamiento de la película',
}

```

A continuación se muestra un diccionario que describe cada columna en el conjunto de datos del archivo ML_data.csv:

```python
column_description = {
    'id': 'ID de la película',
    'title': 'Título de la película',
    'genero': 'Género de la película',
    'popularity': 'Popularidad de la película'
}

```

A continuación se muestra un diccionario que describe cada columna en el conjunto de datos del archivo cast_data.csv:

```python
column_description = {
    'id': 'ID de la película',
    'cast': 'Elenco de la película en formato JSON'
}

```



A continuación se muestra un diccionario que describe cada columna en el conjunto de datos del archivo movie_genres.csv:

```python
column_description = {
    'id': 'ID de la película',
    'id_genres': 'ID de géneros asociados a la película',
    'genero': 'Géneros de la película'
}

```

## Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exploratorio utilizando técnicas estadísticas y visualizaciones. El archivo donde se llevó a cabo este análisis se encuentra en la carpeta "EDA", en un notebook que contiene las gráficas de exploración.

Este análisis exploratorio de datos fue fundamental para comprender mejor el conjunto de datos y a tomar decisiones informadas en etapas posteriores del proyecto. Se descubrieron nuevos patrones, identificar outliers y obtener una comprensión más profunda de las características de las películas y su éxito financiero.

# Sistema de Recomendación

## Desarrollo de Modelos de Machine Learning

Se desarrollo un modelo de Machine Learning para resolver el siguiente desafío:

Sistema de recomendación: Utilizamos técnicas de filtrado colaborativo y/o basado en contenido para construir un sistema de recomendación de películas personalizadas.

Mediante la API, los usuarios pueden ingresar el nombre de una película y el endpoint correspondiente les proporcionará 5 recomendaciones basadas en sus características y en las preferencias de otros usuarios con gustos similares. Esto mejora la experiencia del usuario al ofrecer sugerencias relevantes y personalizadas.

## Análisis del modelo de aprendizaje automático "k vecinos más cercanos"

Para el desarrollo del sistema de recomendación, se proporcionó un enunciado, "Este consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas". El enunciado me pide que encuentre la similitud de puntuación entre una película y las demás películas, y naturalmente, para esto se creara el sistema de recomendación utilizando la función "cosine_similarity".

Para poder realizar una mejor predicción en el modelo, se agrego el género de la película.


# API con FASTAPI

## Desarrollo de la API

La API se desarrolló utilizando FastAPI, un framework web de Python que nos permite crear servicios web de manera rápida y eficiente. A continuación, se mencionan las librerías y frameworks utilizados en la creación de la API, junto con una breve descripción de su función y uso en el proyecto:

- `FastAPI`: FastAPI es un framework web de alto rendimiento basado en Python. Se utilizó para crear y gestionar la API, proporcionando rutas y controladores para las diferentes funciones y endpoints.
- `pandas`: pandas es una librería de Python ampliamente utilizada para la manipulación y análisis de datos. Se utilizó para cargar y procesar los conjuntos de datos, permitiendo realizar consultas y realizar operaciones sobre ellos.
- `sklearn.neighbors`: sklearn.neighbors es un módulo de la librería scikit-learn que contiene algoritmos de vecinos más cercanos (K-Nearest Neighbors). Se utilizó para implementar funcionalidades relacionadas con el sistema de recomendación, como encontrar vecinos más cercanos basados en características similares.

Una vez ejecutados estos comandos, se tendran instaladas las librerías y frameworks necesarios para ejecutar la API y utilizar sus funcionalidades.

# Deployment

## Despliegue (Deployment)

Para el despliegue de la API en Render, se creó un archivo llamado `requirements.txt`. Este archivo es utilizado para especificar las dependencias y las versiones exactas de las librerías que son necesarias para que la API funcione correctamente.

Render utiliza el archivo `requirements.txt` para instalar automáticamente las dependencias especificadas en el entorno de ejecución de la aplicación. Al incluir las dependencias y las versiones adecuadas en este archivo, se asegura que la API pueda ejecutarse sin problemas en Render, con todas las bibliotecas necesarias correctamente instaladas.

Cada línea del archivo especifica el nombre de una librería seguido de `==` y la versión requerida. Pueden incluirse tantas líneas como sean necesarias para todas las dependencias de la API.

Una vez que el archivo `requirements.txt` está correctamente configurado, Render utilizará esta información para instalar automáticamente las librerías necesarias durante el proceso de despliegue de la API. Esto asegura que todas las dependencias estén disponibles en el entorno de ejecución de la aplicación en Render.


---

# Video de Demostración

En este video tutorial, podrán encontrar una descripción paso a paso de cómo utilizar la aplicación, las principales funcionalidades que ofrece y cómo aprovechar al máximo sus características.

<a href="https://youtu.be/vhlR6XJjDKM" target="_blank">Video de Demostración</a>

---

![Texto Alternativo](https://blog.soyhenry.com/content/images/size/w1000/2022/04/Data2_logo.png)
