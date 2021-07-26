# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 02:41:49 2021

@author: Pika
"""

import numpy as np #Algebra lineal
import pandas as pd # Procesamiento de Datos
import matplotlib.pyplot as plt # Visualización
import seaborn as sns # Visualización
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


from pytz import timezone
import pytz


pd.set_option('display.max_columns', 500)

import os
for dirname, _, filenames in os.walk('/Documentos/2021-1/Machine Learning con Python/proyecto'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/Documentos/2021-1/Machine Learning con Python/proyecto/SolarPrediction.csv')


"""
En la parte superior se llama el archivo desde la ubicación de la carpeta,
despues en la parte inferior lo que se puede apreciar es la reorganización
de los datos, eliminando algunas columnas innecesarias y organizando el 
parametro del tiempo con respecto a cada muestreo hecho 

"""



data['Date'] = pd.to_datetime(data['Data']).dt.date.astype(str)
data['TimeSunRise'] = data['Date'] + ' ' + data['TimeSunRise']
data['TimeSunSet'] = data['Date'] + ' ' + data['TimeSunSet']
data['Date'] = data['Date'] + ' ' + data['Time']


data = data.sort_values('Date').reset_index(drop=True)
data.set_index('Date', inplace=True)
data.drop(['Data', 'Time', 'UNIXTime'], axis=1, inplace=True)
data.columns = data.columns.str.lower()
data.index = pd.to_datetime(data.index)



"""
En la parte inferior se aprecia el conocimiento basico de los estadisticos de
los datos y la descripcion de los mismos. Más abajito se organiza en forma de
DataFrame. Esta el HeatMap que tiene la funcion de determinar si se encuentran 
datos nulos o inconsistentes en el conjunto de datos mediante el ploteo de una
grafica de mapa de calor.
"""
#print(data.head())
#print(data.shape)
print(data)
statistics = data.describe()
print(statistics)

"Muestra de datos "

# =============================================================================
# plt.figure(figsize=(14,4))
# sns.heatmap(data.isnull(),cbar=False,cmap='viridis',yticklabels=False)
# plt.title('Busqueda de valores vacios en el DataSet')
# 
# =============================================================================
df = pd.DataFrame(data)
print(df)

"""
En esta parte inferior se procede a determinar como es el comportamiento de
los dato de forma general, se plotean los pairplots, y los scatter plots.
Se evidencia una gran cantidad de datos que ya sobrecargan el analisis que
se esperaba interpretar.
"""

# pair plot of sample feature
#sns.pairplot(df, kind='hist',
#             vars = ['radiation', 'temperature', 'pressure', 'humidity'] )


"Scatter Plots"
# Para Humedad vs Radiación

# =============================================================================
# plt.scatter(data.humidity,data.radiation)
# plt.xlabel("Humedad")
# plt.ylabel("Radiacion")
# 
# =============================================================================

# Para Temperatura vs Radiación
# =============================================================================
# 
# plt.scatter(data.temperature,data.radiation)
# plt.xlabel("Temperatura")
# plt.ylabel("Radiacion")
# 
# 
# =============================================================================


# Para Presión vs Radiación
# =============================================================================
# 
# plt.scatter(data.pressure,data.radiation)
# plt.xlabel("Presión")
# plt.ylabel("Radiación")
# 
# 
# =============================================================================

"Correlación de los datos"


# =============================================================================
# fig = plt.figure()
# fig.suptitle('Correlaciones de las caracteristicas', fontsize=18)
# sns.heatmap(data.corr(), annot=True, cmap='RdBu', center=0)
# 
# =============================================================================



"""
En esta parte inferior se procede a determinar como es el comportamiento de
los dato de forma especifica, se seleccionaron dos dias al azar para
evidenciar relaciones, se plotean los pairplots, y los scatter plots.

"""


datos1 = df.iloc[11152:11439] #10/18/2016 m/d/a
datos2 = df.iloc[21071:21358] #11/12/2016


"Graficas de las relaciones entre muestras, comparacion dos dias seleccionados al azar"


# =============================================================================
# sns.pairplot(datos1,
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# sns.pairplot(datos1, kind='kde',
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# sns.pairplot(datos1, kind='hist',
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# 
# 
# sns.pairplot(datos2,
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# sns.pairplot(datos2, kind='kde',
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# sns.pairplot(datos2, kind='hist',
#              vars = ['radiation', 'temperature', 'pressure', 'humidity'], corner=True )
# 
# =============================================================================

"ScatterPlots"

# =============================================================================
# # Para Humedad vs Radiación
# plt.figure()
# plt.scatter(datos1.humidity,datos1.radiation)
# plt.xlabel("Humedad")
# plt.ylabel("Radiacion")
# plt.title('10/18/2016')
# 
# # Para Temperatura vs Radiación
# plt.figure()
# plt.scatter(datos1.temperature,datos1.radiation)
# plt.xlabel("Temperatura")
# plt.ylabel("Radiacion")
# plt.title('10/18/2016')
# 
# # Para Presión vs Radiación
# plt.figure()
# plt.scatter(datos1.humidity,datos1.temperature)
# plt.xlabel("Humedad")
# plt.ylabel("Temperatura")
# plt.title('10/18/2016')
# 
# # Para Humedad vs Radiación
# plt.figure()
# plt.scatter(datos2.humidity,datos2.radiation)
# plt.xlabel("Humedad")
# plt.ylabel("Radiacion")
# plt.title('11/12/2016')
# 
# # Para Temperatura vs Radiación
# plt.figure()
# plt.scatter(datos2.temperature,datos2.radiation)
# plt.xlabel("Temperatura")
# plt.ylabel("Radiacion")
# plt.title('11/12/2016')
# 
# # Para Presión vs Radiación
# plt.figure()
# plt.scatter(datos2.humidity,datos2.temperature)
# plt.xlabel("Humedad")
# plt.ylabel("Temperatura")
# plt.title('11/12/2016')
# 
# =============================================================================

#==============================================================================
'probabilidad empirica conjunta'

df.info()
df.describe()
corr = df.corr()

df.columns
df1=df[[ 'radiation', 'temperature', 'pressure',
       'humidity', 'winddirection(degrees)', 'speed', 'timesunrise',
       'timesunset']]

h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];


#==============================================================================
'regresion lineal'
from sklearn.model_selection import train_test_split
X = df[['temperature', 'humidity', 'winddirection(degrees)']] #Independent variable 
y = df['radiation'] #dependent variable 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 
lm.fit(X_train,y_train) 
predictions = lm.predict(X_test)
# R_square 
sse = np.sum((predictions - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)

print('R square obtain for normal equation method is :',R_square)

f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,predictions,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - predictions),ax=ax,color='b')
ax.axvline((y_test - predictions).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 
print('R2: ', metrics.r2_score( y_test, predictions))
print('EVS: ', metrics.explained_variance_score(y_test, predictions))


#Regresion Log

print('MSL: ', metrics.mean_squared_log_error(abs(y_test), abs(predictions)))








