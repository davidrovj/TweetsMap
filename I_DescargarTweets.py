# -*- coding: utf-8 -*-
"""Este script permite descargar tweets en tiempo real de una ciudad o zona especificada."""

# In[]: 
"""Paquetes necesarios."""
import json
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

# In[]:
"""Importar los datos de acceso a la API de Twitter, definidos en el script Credentials.py."""
from credentials import *

# In[]:
"""Establecer acceso a la API de Twitter."""
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# In[]:
"""Definir las coordenadas del área del cual se quiere recuperar los tweets.
En esta práctica las ciudades elegidas son Tampico y Ciudad Madero.
Para consultar las coordenadas de una ciudad puede utilizar el siguiente sitio web: https://www.geodatos.net/

* Nota: las coordenadas deben proporcionarse de la siguiente manera:
[sw_longitude, sw_latitude, ne_longitude, ne_latitude]
"""
#---------------------------COORDENADAS_TAMPICO-----------||---------COORDENADAS_MADERO--------------
tampico_madero = [-97.87777, 22.28519, -96.87777, 23.28519, -97.83623, 22.27228, -96.83623, 23.27228]

# In[]:
"""Definir la clase que permitirá el uso del Twitter Streaming API."""
#Twitter Streaming API nos permite descargar mensajes de twitter en tiempo real
class listener(tweepy.StreamListener):
    
    def __init__(self, numero_tweets):
        self.received_tweets_counter = 0
        self.max_number_tweets = numero_tweets
        #Directorio y nombre del archivo de texto donde guardaremos los tweets
        self.file = open('tweets/tweets_tampico_madero.txt', 'a', encoding="UTF-8")
        super(listener, self).__init__()
    
    def on_data(self, data):
        if (self.received_tweets_counter < self.max_number_tweets):
            
            self.received_tweets_counter += 1
            #La API de Twitter devuelve datos en formato JSON,
            #asi que hay que decodificarlos.
            try:
                decoded = json.loads(data)
            except Exception as e:
                print(e)
                return True
            #No todos los usuarios tienen habilitada la opcion de geolocalizacion
            #Por ello hay que dar formato a cuando no este disponible
            if decoded.get('geo') is not None:                
                location = str(decoded.get('geo').get('coordinates'))
            else:
                location = '[,]'
            
            #Extraer los datos que nos interese de los tweets
            text = decoded['text'].replace('\n',' ')
            user = '@' + decoded.get('user').get('screen_name')
            created = decoded.get('created_at')
            
            #Escribir los tweets en el archivo de texto
            self.file.write(user + "|" + location + "|" + created + "|" + text + "\n")
            
            return True
        else:
            self.file.close()
            print('Done!')
            return False

    #def on_status(self, status):
        #print(status.text)
        
    def on_error(self, status):
        print(status)
        #Debido a que el uso de la API de twitter tiene un limite diario debemos
        #desconectarnos del stream cuando excedamos dicho limite
        if status == 420:
            print('status code 420: ' + status)
            self.file.close()
            #Retornando un False en on_error Conseguimos desconectarnos del Stream
            return False
        self.file.close()

# In[]:
"""Comenzar a capturar los tweets en el archivo de texto (tweets_tampico_madero.txt),
definido en la clase anterior.

Salidas:
    En consola:
        [1] Starting…
        [2] Done!
    Archivos:
        [1] El archivo de texto con los tweets (tweets_tampico_madero.txt)

"""
if __name__ == '__main__':
    print('Starting...')
    #Crear un Stream, estableciendo el número de tweets que se desean guardar
    twitterStream = tweepy.Stream(auth, listener(numero_tweets = 47))
    #Iniciar un Stream (Puede tomar mucho tiempo en terminar)
    #Mas info. en: https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/basic-stream-parameters
    twitterStream.filter(locations = tampico_madero)
