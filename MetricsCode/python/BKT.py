
from HMM import hmm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, mean_squared_error

np.random.seed(1000)
folder = "/Users/pedroantonio/Desktop/BKT implementations/Bayesian-Knowledge-Tracing-master/bkt/"   

#Objetivo para la ganancia de aprendizaje
matrizInicial = np.load(folder + "TutorialKnowledgeMatrix.npy") 
matrizIntermediate = np.load(folder + "intermediateKnowledgeMatrix.npy")
print(matrizInicial)
#Crea los conjuntos de train/test
kf = KFold(n_splits=5, shuffle=True)

matrizTrain_pred = []
matrizTrain_actual = []
matrizTest_pred = []
matrizTest_actual = []

#Número de knowledge components (Ahora mismo 1, a la espera de identificar más)
numkc = 1

#Se itera para cada combinación diferente de train test
for train_index, test_index in kf.split(matrizInicial):
    
    #Se divide la matriz inicial en train test
    matrizInicial_train = matrizInicial[train_index]
    matrizInicial_test = matrizInicial[test_index]
    
    matrizIntermediate_train = matrizIntermediate[train_index]
    matrizIntermediate_test = matrizIntermediate[test_index]
    
    # Simbolos: 1 incorrecto, 2 correcto
    symbols = [['1', '2']]
    
    #Objeto HMM pasando como parámetro: 
    # Probabilidad Inicial PI --> P(L0) 1-P(L0)
    # Calculada en base al número de aciertos de los puzzles de tutorial
    # Probabilidad de cambiar de un estado a otro T --> A
    # Probabilidad de Emission E --> B 
    # Símbolos anteriores
    h = hmm(2, Pi=np.array([0.48, 0.52]), T=np.array([[1, 0], [0.4, 0.6]]),E=[np.array([[0.95, 0.05], [0.05, 0.95]])], obs_symbols=symbols)
    #h = hmm(2, Pi=np.array([0.8, 0.2]), T=np.array([[1, 0], [0.4, 0.6]]), obs_symbols=symbols)

    #Inicialización a vacío
    conjunto_train = [[] for x in range(numkc)]
    conjunto_test = [[] for x in range(numkc)]
    
    
    #Iteración para cada kc, en este caso 1 ya que no se ha hecho diferenciación de puzzles 
    for i in range(numkc):
        
        #Matriz con lista de respuestas por alumno
        datosPuzzles = np.load(folder + "IntermediatePuzzlesResults.npy", allow_pickle=True)
        print(datosPuzzles)
        #Separación de matriz de alumnos en train y test
        datosPuzzles_train, datosPuzzles_test = datosPuzzles[train_index], datosPuzzles[test_index]
        
        #Se almacena el train y test para su análisis
        train = [each for each in datosPuzzles_train if each]
        test = [each for each in datosPuzzles_test if each]
        
        #Ajusta los parámetros del modelo para maximizar la probabilidad de la secuencia de observaciones
        if train and test:
            h.baum_welch(train, debug=False)   
        
        #Con el entrenamiento del modelo, se aplica a los datos del puzzle para obtener la probabilidad del siguiente caso
        conjunto_train[i].extend(h.predict_nlg(datosPuzzles_train))
        conjunto_test[i].extend(h.predict_nlg(datosPuzzles_test))

    #print(conjunto_train)
    #print("Tamaño conjunto train: ", len(conjunto_train[0]))
    #print("Conjunto test: ",conjunto_test )
    #print("Tamaño conjunto test: ", len(conjunto_test[0]))    
        
    conjunto_train = np.transpose(conjunto_train)
    conjunto_test = np.transpose(conjunto_test)

    conjunto_train = pd.DataFrame(conjunto_train).fillna(value=0)
    conjunto_test = pd.DataFrame(conjunto_test).fillna(value=0)
    
    
    # Regresión logística para aplicar el modelo en la matriz inicial
    #print("*************Comienza la regresión logística******************")
    logreg = LogisticRegression()
    #Se entrena con las probabilidades calculadas
    logreg.fit(conjunto_train, pd.DataFrame(matrizInicial_train))
    #Se predice el cambio de la matriz inicial con las probabilidades de aprendizaje
    predict = logreg.predict(conjunto_train)
    
    #matrizTrain_pred.extend([each for each in predict])
    matrizTrain_pred = predict
    print(matrizTrain_pred)

    #Se coge la parte de entrenamiento de la matriz inicial
    matrizTrain_actual = matrizInicial_train

    predict = logreg.predict(conjunto_test)
    #matrizTest_pred.extend([each for each in predict])
    matrizTest_pred = predict
    matrizTest_actual = matrizInicial_test
    print(matrizIntermediate)

print (" ")
print ("<<<<<<< Student Learning Gain >>>>>>>")


#########################################################

matrizRes = matrizTrain_pred - matrizTrain_actual
#print(matrizTrain_pred)
#print(matrizTrain_actual)
#print(matrizRes)

cont1=0
cont0=0
contM=0
sumaTotal=0


for i in matrizTrain_pred: 
    sumaTotal = i + sumaTotal
    if( i == 1):
        cont1=cont1+1
    elif(i == 0):
        cont0=cont0+1
    else: contM = contM+1
        
#print("1´s: ", cont1)
#print("0´s: ", cont0)
#print("-1: ", contM)
#print("Suma total: ", sumaTotal)
#print("Learning gain: ", sumaTotal/len(matrizTrain_pred))
#print(" ")


cont1=0
cont0Off=0
contM=0
sumaTotal=0


for i in matrizTrain_actual: 
    sumaTotal = i + sumaTotal
    if( i == 1):
        cont1=cont1+1
    elif(i == 0):
        cont0Off=cont0Off+1
    else: contM = contM+1
        
#print("1´s: ", cont1)
#print("0´s: ", cont0Off)
#print("-1: ", contM)
#print("Suma total: ", sumaTotal)
#print("Learning gain: ", sumaTotal/len(matrizRes))
#print(" ")

cont1=0
cont0=0
contM=0
sumaTotal=0


for i in matrizRes: 
    sumaTotal = i + sumaTotal
    if( i == 1):
        cont1=cont1+1
    elif(i == 0):
        cont0=cont0+1
    else: contM = contM+1
        

print("Learning gain: ", round(sumaTotal/cont0Off, 2))



matrizError = matrizTrain_pred - matrizIntermediate_train
#print(matrizError)
cont = 0
for i in matrizError:
    if(i != 0):
        cont+=1
        
#print(cont/len(matrizError))

########################################################


print ("<<<<<<< Student Modeling >>>>>>> ")
#print ("Training MSE: ", mean_squared_error(matrizTrain_pred,matrizIntermediate_train))
print ("Difference between predicted matrix and matrix of intermediate puzzles: ", mean_squared_error(matrizIntermediate_test, matrizTest_pred))
print(" ")


