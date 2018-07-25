## ENTRENAMIENTO Y PRUEBA DE MODELO DE RANDOM FOREST EN
## PREDICCION DE TIEMPO DE ESTANCIA EN HOSPITAL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#####################################################
# Cargar datos
print('Cargando datos...')
X = pd.read_csv('data_preprocesada.csv')
y= pd.read_csv('y.csv')
feat_labels = ['Diagnostico','Hospital','via_Ingreso','codigo_Administradora','Causa_Externa','Edad','Ocupacion','Num_Reinserciones']


# Separar en datos de entrenamiento y validacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)




######################################################
# Entrenar clasificador
print('Entrenando modelo Random Forest ...')
model_rf = RandomForestClassifier(n_estimators=1000, max_features=4, min_samples_leaf=10,random_state=0, n_jobs=2)
model_rf.fit(X_train, y_train.values.ravel())




#####################################################
# Encontrar importancia de cada variable, y graficar
importanciaVars=model_rf.feature_importances_
print('Calculando importancia de variables para prediccion ...')

# Graficar barras
pos=[1, 2, 3, 4, 5, 6, 7, 8]
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(feat_labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()



######################################################
# Realizar prediccion en datos de validacion
print('Prediccion en datos de validacion...')
y_pred = model_rf.predict(X_test)
precision=accuracy_score(y_test, y_pred)
print(precision)

# Matriz de confusion
tabla=pd.crosstab(y_test.values.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
print(tabla)
