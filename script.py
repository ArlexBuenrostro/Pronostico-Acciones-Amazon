# Importamos las librerias necesarias
import pandas as pd
import numpy as np
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# Función para calcular si la serie es estacionaria o no
def check_stationarity(serie_param):
    result = adfuller(serie_param.dropna())
    p_value = result[1]
    if p_value < 0.05:
        print('\nLa serie es estacionaria.')
    else:
        print('\nLa serie no es estacionaria.')

# Leemos los datos
data = pd.read_csv('Amazon_stock_data.csv')

# Convertimos la fecha a tipo datetime
data['Date'] = pd.to_datetime(data['Date'])

# Indicamos que Date será nuestro índice (así python reconoce que es una serie de tiempo)
data.set_index('Date', inplace=True)

# Indicamos que sólo nos interesan los datos del 2012 al 2018
data = data.loc['2012-01-01':'2018-01-01']

# Remuestreamos pero a meses
data_mensual = data.resample('M').mean()

# Asignamos la columna que será la serie
serie = data_mensual['Close']

# Revisamos si la serie es estacionaria asi normal
check_stationarity(serie)

#Diferenciamos la serie y volvemos a comprobar si ya es estacionaria
serie_diff = serie.diff()
check_stationarity(serie_diff)

# Imprimos las gráficas de ACF y PACF
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plot_acf(serie_diff.dropna(), ax=plt.gca(), lags=24)
plt.title('ACF de la Serie Diferenciada (Mensual)', fontsize=14)
plt.subplot(1, 2, 2)
plot_pacf(serie_diff.dropna(), ax=plt.gca(), lags=24)
plt.title('PACF de la Serie Diferenciada (Mensual)', fontsize=14)
plt.tight_layout()
plt.show()

# Imprimimos las gráficas descomponiendo la serie
decomposition = seasonal_decompose(serie.dropna(), model='multiplicative', period=12)
fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.tight_layout()
plt.show()

# Buscaremos la mejor configuración de SARIMA
mejor_aic = float('inf')
mejor_orden = None
mejor_orden_estacional = None

for p in range(2): 
    for d in range(2):
        for q in range(2):
            for P in range(2):
                for D in range(2):
                    for Q in range(2):
                        orden = (p, d, q)
                        orden_estacional = (P, D, Q, 12)
                            
                        model_fit = SARIMAX(serie, order=orden, seasonal_order=orden_estacional).fit(disp=False)
                        aic = model_fit.aic
                        print(f'SARIMAX{orden}{orden_estacional} -> AIC: {aic:.2f}')

                        if aic < mejor_aic:
                            mejor_aic = aic
                            mejor_orden = orden
                            mejor_orden_estacional = orden_estacional

print(f"\nEl mejor modelo es SARIMAX{mejor_orden}{mejor_orden_estacional} con un AIC de {mejor_aic:.2f}")

# Ajustamos el modelo final con los mejores parámetros
model = SARIMAX(serie, order=mejor_orden, seasonal_order=mejor_orden_estacional)
resultados_finales = model.fit()
print(resultados_finales.summary())

# Predecimos
n_predicciones = 36
predicciones = resultados_finales.forecast(steps=n_predicciones)
print(f"\nPredicción para los próximos {n_predicciones} meses:")
print(predicciones)

# Graficamos los históricos junto a lo pronosticado
plt.figure(figsize=(15, 7))
plt.plot(serie, label='Datos Históricos (Observados)')
plt.plot(predicciones, label='Predicciones (Forecast)', color='red')
plt.title(f'Predicción Mensual con SARIMAX{mejor_orden}{mejor_orden_estacional}', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (Promedio Mensual)')
plt.legend()
plt.grid(True)
plt.show()

# Usaremos el 80% de los datos para entrenar y el 20% para probar
train_size = int(len(serie) * 0.80)
train, test = serie[0:train_size], serie[train_size:len(serie)]

# Entrenar el modelo solo con los datos de entrenamiento
modelo_para_prueba = SARIMAX(train, order=mejor_orden, seasonal_order=mejor_orden_estacional)
resultados_prueba = modelo_para_prueba.fit(disp=False)

# Hacemos predicciones para el mismo periodo que cubren los datos de prueba
predicciones_prueba = resultados_prueba.forecast(steps=len(test))

# Calculamos los errores comparando las predicciones con los datos reales de prueba
rmse = np.sqrt(mean_squared_error(test, predicciones_prueba))
mae = mean_absolute_error(test, predicciones_prueba)

print(f"\nRMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
print(f"MAE (Error Absoluto Medio): {mae:.2f}")
print(f"En promedio, las predicciones del modelo se desvían del valor real en ${mae:.2f}.\n")

# Gráfico con la predicción y los datos reales
plt.figure(figsize=(15, 7))
plt.plot(train, label='Datos de Entrenamiento')
plt.plot(test, label='Datos Reales (Prueba)', color='orange', linewidth=2)
plt.plot(predicciones_prueba, label='Predicciones del Modelo', color='red', linestyle='--')

plt.title('Rendimiento del Modelo SARIMAX vs. Datos Reales de Prueba', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (Promedio Mensual)')
plt.legend()
plt.grid(True)
plt.show()