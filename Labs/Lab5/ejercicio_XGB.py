## ENTRENAMIENTO Y PRUEBA DE MODELO DE XGBOOST EN
## PREDICCION DE TIEMPO DE ESTANCIA EN HOSPITAL

# Hay que instalar paquete xgboost
#conda install -c anaconda py-xgboost-cpu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Escribir aqui el codigo fuente que resuelva el problema