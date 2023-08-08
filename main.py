from sklearn.linear_model import LinearRegression
from datetime import datetime
from fastapi import FastAPI
from fastapi import Path
import pandas as pd
import numpy as np
import ast

with open('Datasets/steam_games.json', 'r') as file:
    data_list = [ast.literal_eval(record) for record in file.readlines()]

data = pd.DataFrame(data_list)

def get_format(date):
    formats_list = ['%Y-%m-%d', '%d.%m.%Y', '%b %Y', '%b-%y', '%B %Y', '%Y', '%d %b, %Y']
    for formats in formats_list:
        try:
            parsed_date = datetime.strptime(date, formats)
            if parsed_date.strftime(formats) == date:
                return parsed_date.date().strftime('%Y-%m-%d')
        except ValueError:
            continue
        except TypeError:
            return np.nan
    return np.nan

data['modify_date'] = data['release_date'].apply(get_format)

data.dropna(subset=['modify_date'], inplace=True, ignore_index=True)
data['modify_date'] = pd.to_datetime(data['modify_date'])
data['year'] = data['modify_date'].dt.year
data['year'] = data['year'].astype(str)


eda_data = pd.read_csv('Datasets/data_predict.csv')

# Se ingresa un año y devuelve un diccionario con los 5 géneros más vendidos en el orden correspondiente.
def get_genre(year):
    filtered_genre = data[data['year'] == str(year)]
    genres = [genre for lista in filtered_genre['genres'] if isinstance(lista, list) for genre in lista if
              pd.notna(genre)]
    genres_dict = {genre: genres.count(genre) for genre in genres}
    top_genres = dict(sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_genres


# Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
def get_games(year):
    filtered_games = data[data['year'] == str(year)]
    games_list = [game for game in filtered_games['app_name'].unique() if pd.notna(game)]
    games_dict = {year: games_list}
    return games_dict


#  Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
def get_specs(year):
    filtered_specs = data[data['year'] == str(year)]
    specs = [spec for lista in filtered_specs['specs'] if isinstance(lista, list) for spec in lista if pd.notna(spec)]
    specs_dict = {spec: specs.count(spec) for spec in specs}
    top_specs = dict(sorted(specs_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_specs


#  Cantidad de juegos lanzados en un año con early access
def get_early_access(year):
    filtered_early_access = data[data['year'] == str(year)]
    early_access_count = filtered_early_access[filtered_early_access['early_access'] == True].shape[0]
    early_access_dict = {year: early_access_count}
    return early_access_dict


# Según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
def get_sentiment(year):
    sentiments = ['Mixed', 'Mostly Negative', 'Mostly Positive', 'Negative', 'Overwhelmingly Negative',
                  'Overwhelmingly Positive', 'Positive', 'Very Negative', 'Very Positive']
    filtered_sentiment = data[data['year'] == str(year)]
    filtered_sentiment = filtered_sentiment[filtered_sentiment['sentiment'].isin(sentiments)]
    sentiment_list = [x for x in filtered_sentiment['sentiment'] if pd.notna(x)]
    sentiment_dict = {sentiment: sentiment_list.count(sentiment) for sentiment in sentiment_list}
    return dict(sorted(sentiment_dict.items(), key=lambda x: x[1], reverse=True))


# Top 5 juegos, según año, con mayor metascore.
def get_metascore(year):
    filtered_metascore = data[(data['year'] == str(year)) & (data['metascore'] != 'NA')]
    filtered_metascore = filtered_metascore.dropna(subset=['metascore'])
    sorted_metascore = filtered_metascore.sort_values('metascore', ascending=False)
    top_data = sorted_metascore.head(5)[['app_name', 'metascore']]
    if top_data.empty:
        metascore_dict = {year: 'Data not found'}
    else:
        metascore_dict = top_data.set_index('app_name')['metascore'].to_dict()
    return metascore_dict

def predict_price(genre, early_access, year, sentiment):
    # Codificar el género como binario
    eda_data['genre_encoded'] = eda_data['genres'].apply(lambda x: genre in x)

    # Convertir el sentimiento en valores numéricos
    sentiment_mapping = {'Mixed': 0, 'Very Positive': 1, 'Positive': 2, 'Mostly Positive': 3,'Mostly Negative': 4, 'Overwhelmingly Positive': 5, 'Negative': 6, 'Very Negative': 7, 'Overwhelmingly Negative': 8}
    sentiment_numeric = sentiment_mapping[sentiment]

    # Crear la columna 'sentiment_numeric' en eda_data
    eda_data['sentiment_numeric'] = eda_data['sentiment'].apply(lambda x: sentiment_mapping[x])

    # Crear el conjunto de características (X)
    X = eda_data[['genre_encoded', 'year', 'early_access', 'sentiment_numeric']]

    # Crear el conjunto de etiquetas (y)
    y = eda_data['price']

    # Crear y ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Crear un diccionario con los valores de las características para la predicción
    input_data = {
        'genre_encoded': genre in eda_data['genres'],
        'year': year,
        'early_access': early_access,
        'sentiment_numeric': sentiment_numeric
    }

    # Crear una fila de datos a partir del diccionario
    input_row = pd.DataFrame([input_data])

    # Realizar la predicción
    y_predict = model.predict(input_row)

    # Calcular el RMSE a partir del MSE
    rmse = np.sqrt(np.mean((y_predict - eda_data['price'])**2))

    result = {'price': y_predict[0], 'RMSE': rmse}
    return result


app = FastAPI(title='FastAPI by Bianca Torres')


@app.get('/')
def read_root():
    return {"Hello": "Welcome"}


@app.get('/genero/{year}', description= 'Se ingresa un año y devuelve un diccionario con los 5 géneros más vendidos en el orden correspondiente.')
def genero(year: str = Path(..., description="Elige el año", enum=[x for x in data.year.unique()])):
    return get_genre(year)

@app.get('/juegos/{year}', description= 'Se ingresa un año y devuelve un diccionario con los juegos lanzados en el año.')
def juegos(year: str = Path(..., description="Elige el año", enum=[x for x in data.year.unique()])):
    return get_games(year)


@app.get('/specs/{year}', description= 'Se ingresa un año y devuelve un diccionario con los 5 specs que más se repiten en el mismo en el orden correspondiente.')
def specs(year: str= Path(..., description="Elige el año", enum=[x for x in data.year.unique()]) ):
    return get_specs(year)


@app.get('/earlyaccess/{year}', description= ' Se ingresa un año y devuelve la cantidad de juegos lanzados con early access.')
def earlyacces(year: str = Path(..., description="Elige el año", enum=[x for x in data.year.unique()])):
    return get_early_access(year)


@app.get('/sentiment/{year}', description= 'Según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.')
def sentiment(year: str = Path(..., description="Elige el año", enum=[x for x in data.year.unique()])):
    return get_sentiment(year)


@app.get('/metascore/{year}', description= 'Según el año de lanzamiento, devuelve el Top 5 juegos con mayor metascore.')
def metascore(year: str = Path(..., description="Elige el año", enum=[x for x in data.year.unique()])):
    return get_metascore(year)


@app.get('/prediccion/')
def predicción(genero,earlyaccess,year, sentiment):
    return predict_price(genero, earlyaccess, year, sentiment)
