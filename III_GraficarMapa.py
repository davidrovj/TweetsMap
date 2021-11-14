# -*- coding: utf-8 -*-
"""Este script permite graficar las coordenadas del archivo (tweets_heatmap_tam_mad) en un mapa,
con ayuda del script heatmap.py
"""
# In[]:
"""Paquetes necesarios."""
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
# In[]:
"""Graficar las coordenadas en un mapa.
Salidas:
    En consola:
        [1] Fetching tiles: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 142.70tile/s]
    Archivos:
        [1] una imagen con las coordenadas graficadas en un mapa (map_tweets_tam_mad_01.png)
        [2] una imagen con las coordenadas graficadas en un mapa (map_tweets_tam_mad_02.png)
        [3] una imagen con las coordenadas graficadas en un mapa (map_tweets_tam_mad_03.png)
"""
"""
OPCION 1: Usando el IDE Spyder, Jupyter Qt console o Jupyter notebook.
- Simplemente ejecutar las siguientes lineas

* Información sobre los comandos de heatmap.py en: http://www.sethoscope.net/heatmap/
"""
#[Mapa con nombres]
get_ipython().system('python heatmap.py -o maps/map_tweets_tam_mad_01.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base http://tile.memomaps.de/tilegen --decay 0.8 -r 10 --zoom 0 --margin 15')
#[Mapa Obscuro]
get_ipython().system('python heatmap.py -o maps/map_tweets_tam_mad_02.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/dark_all --decay 0.8 -r 10 --zoom 0 --margin 15')
#[Mapa Claro]
get_ipython().system('python heatmap.py -o maps/map_tweets_tam_mad_03.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15')

"""
OPCION 2: Usando el simbolo del sistema
- Abrir el simbolo del sistema.
- Navegar hasta el directorio del proyecto.
- Ejecutar los siguientes comandos.

* Informacion sobre los comandos de heatmap.py en: http://www.sethoscope.net/heatmap/
"""
#[Mapa con nombres]
##python heatmap.py -o maps/map_tweets_tam_mad_01.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base http://tile.memomaps.de/tilegen --decay 0.8 -r 10 --zoom 0 --margin 15
#[Mapa Obscuro]
#python heatmap.py -o maps/map_tweets_tam_mad_02.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/dark_all --decay 0.8 -r 10 --zoom 0 --margin 15
#[Mapa Claro]
#python heatmap.py -o maps/map_tweets_tam_mad_03.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/light_all --decay 0.8 -r 10 --zoom 0 --margin 15

# In[]:
"""Agregar una leyenda al mapa de los tweets.
como el nombre de la ciudad, la cantidad de tweets graficados y la fecha.

Salidas:
    [1] La imagen de los puntos graficados en un mapa (map_tweets_tam_mad_01.png),
        pero ahora con el texto que agregamos
"""
im = PIL.Image.open('maps/map_tweets_tam_mad_01.png')
draw = ImageDraw.Draw(im)
font = ImageFont.truetype("fonts/ProductSans.ttf", 14)
draw.text((1020, 600),"Tampico-Madero, Tam., Mex.", fill = "black", font=font)
draw.text((1080, 620),"43 tweets 24/07/20", fill = "black", font=font)

im.save('maps/map_tweets_tam_mad_01.png')

# In[]:
"""Definir un nuevo gradiente de color que sustituya a 
los puntos amarillos que marca heatmap.py por defecto. 

Salidas:
    Archivos:
        [1] Gradiente del color que definimos (gradient_blue.png)
"""

hsva_min = Color()
hsva_min.hex_l = '#008dcd'
hsva_max = Color()
hsva_max.hex_l = '#4dccff'
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

imageio.imwrite('gradients/gradient_blue.png', color_gradient)

# In[]:
"""Graficar las coordenadas utilizando el gradiente de color definido anteriormente.

Salidas:
    En consola:
        [1] Fetching tiles: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 142.70tile/s]
    Archivos:
        [1] Un mapa con el nuevo color de puntos (map_tweets_tam_mad_02.png)
"""

"""
OPCION 1: Usando el IDE Spyder, Jupyter Qt console o Jupyter notebook.
- Simplemente ejecutar la siguiente linea
"""
get_ipython().system('python heatmap.py -G gradients/gradient_blue.png -o maps/map_tweets_tam_mad_02.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/dark_all --decay 0.8 -r 10 --zoom 0 --margin 15')

"""
OPCION 2: Usando el simbolo del sistema.
- Abrir el simbolo del sistema.
- Navegar hasta el directorio del proyecto.
- Ejecutar el siguiente comando.
"""
#python heatmap.py -G gradients/gradient_blue.png -o maps/map_tweets_tam_mad_02.png --width 1920 -p heatmap-files/tweets_heatmap_tam_mad -b black -P equirectangular --osm --osm_base  https://basemaps.cartocdn.com/rastertiles/dark_all --decay 0.8 -r 10 --zoom 0 --margin 15

im = PIL.Image.open('maps/map_tweets_tam_mad_02.png')
draw = ImageDraw.Draw(im)
font = ImageFont.truetype("fonts/ProductSans.ttf", 14)
draw.text((1020, 600),"Tampico-Madero, Tam., Mex.", fill = "white", font=font)
draw.text((1080, 620),"43 tweets 24/07/20", fill = "white", font=font)

im.save('maps/map_tweets_tam_mad_02.png')