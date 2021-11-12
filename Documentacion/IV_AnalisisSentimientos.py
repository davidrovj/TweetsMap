# -*- coding: utf-8 -*-
"""En este script se predice si los tweets son positivos o negativos, 
mediante machine learning, para posteriormente ser graficados."""
# In[]:
"""Paquetes Necesarios."""
import PIL
import osmviz
from PIL import Image
from PIL import ImageDraw, ImageFont
from IPython.display import Image
from IPython import get_ipython
import numpy as np
import imageio
from colour import Color
from copy import deepcopy
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
import sklearn
from sklearn.feature_extraction.text import CountVectorizer       
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import langid
from langdetect import detect
import textblob

"""Descargar los dataset que se usarán para predecir los sentimientos de los tweets.
The Spanish Society for Natural Language Processing (SEPLN) proporciona varios dataset 
para el análisis de sentimientos de forma gratuita para propósitos académicos.

Para poder descargar los dataset debe registrarse en el siguiente enlace:
http://tass.sepln.org/tass_data/download.php

De cualquier forma, en la carpeta TASS-Dataset ya se encuentran descargados algunos archivos 
para el análisis de sentimientos.
"""

# In[]:
"""Convertir cada dataset descargado a un archivo CSV y a un data frame."""

pd.set_option('max_colwidth',1000)

# Dataset 1
try:
    general_tweets_corpus_train = pd.read_csv('TASS-Dataset/general-train-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/general-train-tagged.xml', encoding="UTF-8"))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_train = pd.DataFrame(columns=('content', 'polarity', 'agreement'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [tweet.content.text, tweet.sentiments.polarity.value.text, tweet.sentiments.polarity.type.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_train = general_tweets_corpus_train.append(row_s)
    general_tweets_corpus_train.to_csv('TASS-Dataset/general-train-tagged.csv', index=False, encoding='utf-8')

# Dataset 2
try:
    general_tweets_corpus_test = pd.read_csv('TASS-Dataset/general-test-tagged.csv', encoding='utf-8')
except:
    
    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/general-test-tagged.xml', encoding="UTF-8"))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [tweet.content.text, tweet.sentiments.polarity.value.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_test = general_tweets_corpus_test.append(row_s)
    general_tweets_corpus_test.to_csv('TASS-Dataset/general-test-tagged.csv', index=False, encoding='utf-8')

# Dataset 3
try:
    stompol_tweets_corpus_train = pd.read_csv('TASS-Dataset/stompol-train-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/stompol-train-tagged.xml', encoding = "UTF-8"))
    #sample tweet object
    root = xml.getroot()
    stompol_tweets_corpus_train = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [' '.join(list(tweet.itertext())), tweet.sentiment.get('polarity')]))
        row_s = pd.Series(row)
        row_s.name = i
        stompol_tweets_corpus_train = stompol_tweets_corpus_train.append(row_s)
    stompol_tweets_corpus_train.to_csv('TASS-Dataset/stompol-train-tagged.csv', index=False, encoding='utf-8')

# Dataset 4
try:
    stompol_tweets_corpus_test = pd.read_csv('TASS-Dataset/stompol-test-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/stompol-test-tagged.xml', encoding = "UTF-8"))
    #sample tweet object
    root = xml.getroot()
    stompol_tweets_corpus_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [' '.join(list(tweet.itertext())), tweet.sentiment.get('polarity')]))
        row_s = pd.Series(row)
        row_s.name = i
        stompol_tweets_corpus_test = stompol_tweets_corpus_test.append(row_s)
    stompol_tweets_corpus_test.to_csv('TASS-Dataset/stompol-test-tagged.csv', index=False, encoding='utf-8')

# Dataset 5
try:
    social_tweets_corpus_test = pd.read_csv('TASS-Dataset/socialtv-test-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/socialtv-test-tagged.xml', encoding = "UTF-8"))
    #sample tweet object
    root = xml.getroot()
    social_tweets_corpus_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [' '.join(list(tweet.itertext())), tweet.sentiment.get('polarity')]))
        row_s = pd.Series(row)
        row_s.name = i
        social_tweets_corpus_test = social_tweets_corpus_test.append(row_s)
    social_tweets_corpus_test.to_csv('TASS-Dataset/socialtv-test-tagged.csv', index=False, encoding='utf-8')

# Dataset 6
try:
    social_tweets_corpus_train = pd.read_csv('TASS-Dataset/socialtv-train-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/socialtv-train-tagged.xml', encoding = "UTF-8"))
    #sample tweet object
    root = xml.getroot()
    social_tweets_corpus_train = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [' '.join(list(tweet.itertext())), tweet.sentiment.get('polarity')]))
        row_s = pd.Series(row)
        row_s.name = i
        social_tweets_corpus_train = social_tweets_corpus_train.append(row_s)
    social_tweets_corpus_train.to_csv('TASS-Dataset/socialtv-train-tagged.csv', index=False, encoding='utf-8')

# Dataset 7
try:
    general_tweets_corpus_train_mx = pd.read_csv('TASS-Dataset/general-train-tagged-mx.csv', encoding='utf-8')
except:
    
    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/general-train-tagged-mx.xml', encoding="UTF-8"))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_train_mx = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [tweet.content.text, tweet.sentiment.polarity.value.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_train_mx = general_tweets_corpus_train_mx.append(row_s)
    general_tweets_corpus_train_mx.to_csv('TASS-Dataset/general-train-tagged-mx.csv', index=False, encoding='utf-8')

# Dataset 8
try:
    general_tweets_corpus_test_mx = pd.read_csv('TASS-Dataset/general-test-tagged-mx.csv', encoding='utf-8')
except:
    
    from lxml import objectify
    xml = objectify.parse(open('TASS-Dataset/general-test-tagged-mx.xml', encoding="UTF-8"))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_test_mx = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [tweet.content.text, tweet.sentiment.polarity.value.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_test_mx = general_tweets_corpus_test_mx.append(row_s)
    general_tweets_corpus_test_mx.to_csv('TASS-Dataset/general-test-tagged-mx.csv', index=False, encoding='utf-8')

# In[]:
"""Concatenar los dataset anteriores en un solo data frame corpus."""

tweets_corpus = pd.concat([
        social_tweets_corpus_train,
        social_tweets_corpus_test,
        stompol_tweets_corpus_test,
        stompol_tweets_corpus_train,
        general_tweets_corpus_test,
        general_tweets_corpus_train,
        general_tweets_corpus_test_mx,
        general_tweets_corpus_train_mx
        
    ])

# In[]:
"""Modificar el dataframe corpus (tweets_corpus) para que descarte los tweets con polaridad neutral.
El data frame corpus (tweets_corpus) tiene un nuevo campo, por tweet, llamado ‘agreement’, 
el cual solo puede tener dos posibles valores ‘AGREEMENT’ Y ‘DISAGREEMENT’. 
El valor ‘DISAGREEMENT’ corresponde a tweets con polaridad neutral.
"""

tweets_corpus = tweets_corpus.query('agreement != "DISAGREEMENT" and polarity != "NONE"')


# In[]:
"""Definir las funciones para la tokenización y stemming."""

# Eliminar enlaces
tweets_corpus = tweets_corpus[-tweets_corpus.content.str.contains('^http.*$')]
tweets_corpus.shape

spanish_stopwords = stopwords.words('spanish')

non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))
non_words

stemmer = SnowballStemmer('spanish')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # Eliminar caracteres que no sean letras
    text = ''.join([c for c in text if c not in non_words])
    # tokenize
    tokens =  word_tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

# In[]:
"""Preparar el modelo que evaluará los tweets.
Agregar una nueva columna al data frame corpus (tweets_corpus)
en donde las polaridades de los tweets serán representadas de forma binaria.
"""

tweets_corpus = tweets_corpus[tweets_corpus.polarity != 'NEU']
tweets_corpus['polarity_bin'] = 0
tweets_corpus.polarity_bin[tweets_corpus.polarity.isin(['P', 'P+'])] = 1
tweets_corpus.polarity_bin.value_counts(normalize=True)

# In[]:
"""Definir el modelo de evaluación, 
realizar un GridSearch para encontrar los hiperparámetros óptimos y guardar el modelo.

Salidas:
    Archivos:
        [1] El modelo (grid_search.pkl)

Debido a la enorme cantidad de datos en el data frame corpus (tweets_corpus), el modelo puede 
tomar mucho tiempo en terminar, si desea omitir este proceso el modelo ya se encuentra 
en la carpeta del proyecto. Solo debe cargarlo de la siguiente manera:
>>> grid_search = joblib.load('grid_search.pkl')
"""

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])

parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20,50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}

grid_search = sklearn.model_selection.GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search.fit(tweets_corpus.content, tweets_corpus.polarity_bin)

grid_search.best_params_

# Guardar el modelo
joblib.dump(grid_search, 'grid_search.pkl')
# Cargar el modelo
#grid_search = joblib.load('grid_search.pkl')

# In[]:
"""Hacer una validación cruzada para mostrar el rendimiento del modelo.

Salidas:
    En consola:
        [1] 0.9168214861317802
"""

model = LinearSVC(C=.2, loss='squared_hinge',max_iter=1000,multi_class='ovr',
              random_state=None,
              penalty='l2',
              tol=0.0001
)

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = spanish_stopwords,
    min_df = 50,
    max_df = 1.9,
    ngram_range=(1, 1),
    max_features=1000
)

corpus_data_features = vectorizer.fit_transform(tweets_corpus.content)
corpus_data_features_nd = corpus_data_features.toarray()

scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(tweets_corpus)],
    y=tweets_corpus.polarity_bin,
    scoring='roc_auc',
    cv=5
    )
scores.mean()

# In[]:
"""Convertir el archivo CSV con los tweets (tweets_tampico_madero.csv) en un dataframe 
y seleccionar aquellos que se encuentren dentro del área que nos interesa.
"""

tweets = pd.read_csv('tweets/tweets_tampico_madero.csv', names = ["timestamp", "user", "tweet", "lat", "lon"],encoding='utf-8')

tweets = tweets[tweets.tweet.str.len() < 150]
tweets.lat = pd.to_numeric(tweets.lat, errors='coerce')
tweets = tweets[tweets.lat.notnull()]

#Latitudes y Longitudes maximas y minimas del área que nos interesa
min_lon = -97.99
min_lat = 22.20
max_lon = -96.0
max_lat = 23.30

tweets = tweets[(tweets.lat.notnull()) & (tweets.lon.notnull())]
tweets = tweets[(tweets.lon > min_lon) & (tweets.lon < max_lon) & (tweets.lat > min_lat) & (tweets.lat < max_lat)]
tweets.shape

# In[]:
"""Detectar el idioma de los tweets, para posteriormente guardar aquellos que estén en español. """

def langid_safe(tweet):
    try:
        return langid.classify(tweet)[0]
    except Exception as e:
        pass
        
def langdetect_safe(tweet):
    try:
        return detect(tweet)
    except Exception as e:
        pass

def textblob_safe(tweet):
    try:
        return textblob.TextBlob(tweet).detect_language()
    except Exception as e:
        pass   

# Puede tomar mucho tiempo en terminar.
tweets['lang_langid'] = tweets.tweet.apply(langid_safe)
tweets['lang_langdetect'] = tweets.tweet.apply(langdetect_safe)
tweets['lang_textblob'] = tweets.tweet.apply(textblob_safe)
tweets['lang_textblob'] = tweets.tweet.apply(textblob_safe)


# In[]:
"""Exportar el data frame de los tweets con el campo de identificación de idioma
a un archivo CSV (tweets_tampico_madero_2.csv).

Salidas:
    Archivos:
        [1] Un archivo CSV de los tweets con su identificador de idioma
"""

tweets.to_csv('heatmap-files/tweets_tampico_madero_2.csv', encoding='utf-8')

# In[]:
"""-Modificar el data frame de los tweets para solo guardar aquellos que estén en idioma español."""

tweets = tweets.query(''' lang_langdetect == 'es' or lang_langid == 'es' or lang_textblob == 'es'  ''')
tweets.shape

# In[]:
"""Utilizar el modelo entrenado para predecir los sentimientos de lo tweets y 
exportar el data frame resultante a un archivo CSV (tweets_polarity_bin.csv) 
y a un archivo de texto (tweets_heatmap_polarity_binary) para ser graficado posteriormente.

Salidas:
    Archivos:
        [1] Un archivo de texto con las coordenas y la polaridades binarias de los tweets (tweets_heatmap_polarity_binary)
        [2] Un archivo CSV con las coordenas y la polaridades binarias de los tweets (tweets_polarity_bin.csv)
"""

pipeline = Pipeline([
    ('vect', CountVectorizer(
            analyzer = 'word',
            tokenizer = tokenize,
            lowercase = True,
            stop_words = spanish_stopwords,
            min_df = 50,
            max_df = 1.9,
            ngram_range=(1, 1),
            max_features=1000
            )),
    ('cls', LinearSVC(C=.2, loss='squared_hinge',max_iter=1000,multi_class='ovr',
             random_state=None,
             penalty='l2',
             tol=0.0001
             )),
])

pipeline.fit(tweets_corpus.content, tweets_corpus.polarity_bin)
tweets['polarity'] = pipeline.predict(tweets.tweet)

tweets[['tweet', 'lat', 'lon', 'polarity']].to_csv('heatmap-files/tweets_polarity_bin.csv', encoding='utf-8')

with open('heatmap-files/tweets_heatmap_polarity_binary','w') as file:
    file.write(tweets[['lat','lon', 'polarity']].to_string(header=False, index=False))


# In[ ]:
"""Definir dos nuevos gradientes de color para identificar
a los tweets positivos (verde) y los negativos (rojo).
"""

# Verde = tweets positivos
hsva_min = Color()
hsva_min.hex_l = '#24b736'
hsva_max = Color()
hsva_max.hex_l = '#24b736'

color_gradient = list(hsva_max.range_to(hsva_min,256))
alpha = np.arange(0,256)[::-1]
gradient = []

for i, color_point in enumerate(color_gradient):
    rgb = list(color_point.get_rgb())
    rgb = [int(e * 255) for e in rgb]
    rgb.append(alpha[i])
    gradient.append([rgb])

color_gradient = np.array(gradient)
width = 43
color_gradient_row = deepcopy(color_gradient)

for col in range(width-1):
    color_gradient = np.hstack((color_gradient, color_gradient_row))

imageio.imwrite('gradients/gradient_green.png', color_gradient)

# Rojo = tweets negativos
hsva_min = Color()
hsva_min.hex_l = '#ff3639'
hsva_max = Color()
hsva_max.hex_l = '#ff3639'

color_gradient = list(hsva_max.range_to(hsva_min,256))
alpha = np.arange(0,256)[::-1]
gradient = []

for i, color_point in enumerate(color_gradient):
    rgb = list(color_point.get_rgb())
    rgb = [int(e * 255) for e in rgb]
    rgb.append(alpha[i])
    gradient.append([rgb])

color_gradient = np.array(gradient)
width = 43
color_gradient_row = deepcopy(color_gradient)

for col in range(width-1):
    color_gradient = np.hstack((color_gradient, color_gradient_row))

imageio.imwrite('gradients/gradient_red.png', color_gradient)

# In[]:
"""Convertir el archivo de texto de las coordenadas y polaridades (tweets_heatmap_polarity_binary)
en un data frame.
"""

tweets = pd.read_table('heatmap-files/tweets_heatmap_polarity_binary', encoding='utf-8', sep=' ', header=None)
del tweets[tweets.columns[0]]
del tweets[tweets.columns[2]]
tweets.columns = ['lat', 'lon', 'polarity']

# In[]:
"""Exportar las coordenadas con polaridad negativa a un archivo de texto."""

with open('heatmap-files/tweets_with_polarity_negative','w') as file:
    file.write(tweets[['lat','lon']][tweets.polarity==0].to_string(header=False, index=False))

# In[]:
"""Exportar las coordenadas con polaridad positiva a un archivo de texto."""

with open('heatmap-files/tweets_with_polarity_positive','w') as file:
    file.write(tweets[['lat','lon']][tweets.polarity==1].to_string(header=False, index=False))

# In[]:
"""Graficar la capa de tweets negativos.

Salidas:
    En consola:
        [1] Fetching tiles: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 142.70tile/s]
    Archivos:
        [1] Los puntos graficados en un fondo transparente (tampico_madero_tweets_polarity_negative.png)
"""

"""
OPCION 1: Usando el IDE Spyder, Jupyter Qt console o Jupyter notebook.
- Simplemente ejecutar la siguiente linea
"""
get_ipython().system('python heatmap.py -o maps/tampico_madero_tweets_polarity_negative.png --width 1920 -p heatmap-files/tweets_with_polarity_negative -P equirectangular --osm --osm_base https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15 -G gradients/gradient_red.png --layer')

"""
OPCION 2: Usando el simbolo del sistema.
- Abrir el simbolo del sistema.
- Navegar hasta el directorio del proyecto.
- Ejecutar el siguiente comando.
"""
python heatmap.py -o maps/tampico_madero_tweets_polarity_negative.png --width 1920 -p heatmap-files/tweets_with_polarity_negative -P equirectangular --osm --osm_base https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15 -G gradients/gradient_red.png --layer
# In[]:
"""Graficar la capa de tweets positivos.

Salidas:
    En consola:
        [1] Fetching tiles: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 142.70tile/s]
    Archivos:
        [1] Un mapa con los tweets positivos (tampico_madero_tweets_polarity_positive.png)
"""

"""
OPCION 1: Usando el IDE Spyder, Jupyter Qt console o Jupyter notebook.
- Simplemente ejecutar la siguiente linea
"""
get_ipython().system('python heatmap.py -o maps/tampico_madero_tweets_polarity_positive.png --width 1920 -p heatmap-files/tweets_with_polarity_positive -P equirectangular --osm --osm_base https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15 -G gradients/gradient_green.png')

"""
OPCION 2: Usando el simbolo del sistema.
- Abrir el simbolo del sistema.
- Navegar hasta el directorio del proyecto.
- Ejecutar el siguiente comando.
"""
python heatmap.py -o maps/tampico_madero_tweets_polarity_positive.png --width 1920 -p heatmap-files/tweets_with_polarity_positive -P equirectangular --osm --osm_base https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15 -G gradients/gradient_green.png

# In[]:
"""Combinar las imágenes de los tweets postitvos y negativos en un solo archivo.

Salidas:
    Archivos:
        [1] Un mapa con los tweets positivos y negativos
"""

background = PIL.Image.open('maps/tampico_madero_tweets_polarity_positive.png')
foreground = PIL.Image.open('maps/tampico_madero_tweets_polarity_negative.png')

background.paste(foreground, (0, 0), foreground)

draw = ImageDraw.Draw(background)
font = ImageFont.truetype("fonts/ProductSans.ttf", 14)
draw.text((1020, 600),"Tampico-Madero, Tam., Mex.", fill = "black", font=font)
draw.text((1020, 620),"Tweets positivos y negativos", fill = "black", font=font)

background.save('maps/tampico_madero_tweets_polarity.png')