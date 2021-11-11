# -*- coding: utf-8 -*-
"""Este script convierte los tweets del archivo de texto (tweets_tampico_madero.txt) 
a un data frame y a un archivo CSV (tweets_tampico_madero.csv), para su posterior análisis y manipulación. 
También genera un archivo (tweets_heatmap_tam_mad) que contiene solamente las latitudes y longitudes de 
cada tweet.
"""

# In[]:
"""Paquetes necesarios."""
import pandas as pd
import numpy as np

# In[]:
"""Leer el archivo de texto (tweets_tampico_madero.txt) y 
definir un data frame para almacenar los tweets de ese archivo.
"""

tweets_raw = pd.read_table('tweets/tweets_tampico_madero.txt', header=None, iterator=True)
tweets_2 = pd.DataFrame()

# In[]:
"""Convertir los datos del archivo de texto (tweets_tampico_madero.txt) a un data frame y 
a un archivo CSV (tweets_tampico_madero.csv)

Salidas:
    En consola:
        [1] StopIteration
    Archivos:
        [2] Archivo CSV con los tweets (tweets_tampico_madero.csv)
"""
while 1:
    #Debido a que el archivo de texto de los tweets es demasiado grande
    #debe de ser procesado por chunks en lugar de cargar todo el archivo en memoria
    tweets = tweets_raw.get_chunk(1000) #1000 filas por chunk
    tweets.columns = ['tweets']
    tweets['len'] = tweets.tweets.apply(lambda x: len(x.split('|')))
    tweets[tweets.len < 4] = np.nan
    del tweets['len']
    tweets = tweets[tweets.tweets.notnull()]
    #Establecer las columnas y valores que tomarán
    tweets['user'] = tweets.tweets.apply(lambda x: x.split('|')[0])
    tweets['geo'] = tweets.tweets.apply(lambda x: x.split('|')[1])
    tweets['timestamp'] = tweets.tweets.apply(lambda x: x.split('|')[2])
    tweets['tweet'] = tweets.tweets.apply(lambda x: x.split('|')[3])
    tweets['lat'] = tweets.geo.apply(lambda x: x.split(',')[0].replace('[',''))
    tweets['lon'] = tweets.geo.apply(lambda x: x.split(',')[1].replace(']',''))
    del tweets['tweets']
    del tweets['geo']
    #Convertir las latitudes y longitudes de string a float
    tweets['lon'] = pd.to_numeric(tweets['lon'], downcast="float")
    tweets['lat'] = pd.to_numeric(tweets['lat'], downcast="float")
    #Cambiar la zona horaria de UTC a GMT-5
    tweets['timestamp'] = pd.to_datetime(tweets['timestamp'], utc = False)
    tweets = tweets.set_index('timestamp').tz_convert('America/Mexico_City').reset_index()
    #Almacenar los tweets en el dataframe
    tweets_2 = tweets_2.append(tweets, ignore_index = True)
    #Guardar los tweets en un archivo CSV
    tweets.to_csv('tweets/tweets_tampico_madero.csv', mode='a', header=False,index=False)

# In[]:
"""Modificar el data frame con los tweets, para almacenar solo aquellos que 
tengan coordenadas y que a su vez se encuentren en el rango del área que deseamos.
"""
#Latitudes y Longitudes maximas y minimas del area que nos interesa
min_lon = -97.99
min_lat = 22.20
max_lon = -96.0
max_lat = 23.30

tweets_2 = tweets_2[(tweets_2.lat.notnull()) & (tweets_2.lon.notnull())]
tweets_2 = tweets_2[(tweets_2.lon >= min_lon) & (tweets_2.lon <= max_lon) & (tweets_2.lat >= min_lat) & (tweets_2.lat <= max_lat)]

# In[]:
"""Guardar las coordenadas de los tweets en un archivo.

Salidas:
    Archivos:
        [1] Un archivo de texto con las coordenadas de los tweets (tweets_heatmap_tam_mad)
"""

with open('heatmap-files/tweets_heatmap_tam_mad','w') as file:
    file.write(tweets_2[['lat','lon']].to_string(header=False, index=False))