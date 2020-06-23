#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import math
import csv
import os

#inputfolder = '/Users/pedroantonio/Desktop/ELO/Datos Shadowspect'
#datafile = 'datos_ELO.csv'
#trainfile = 'datos_ELO_train.csv'
#testfile = 'datos_ELO_test.csv'

#datafile = 'prueba.csv'
#trainfile = 'pruebaTrain.csv'
#testfile = 'pruebaTest.csv'

student_id = 'user'
student_column_number = 1
group_column_number = 0
completed = 'n_completed'
puzzle_name = 'task_id'
puzzle_column_number = 2
kc_column = 'kc'
kc_column_number = 4

mg1Puzzles = ['Bird Fez', 'Pi Henge', 'Bull Market']
gmd4Puzzles = ['Angled Silhouettes', 'Not Bird', 'Stranger Shapes', 'Ramp Up and Can It', 'Few Clues']
co5Puzzles = ['45-Degree Rotations', 'Boxes Obscure Spheres', 'More Than Meets the Eye']
co6Puzzles = ['Tall and Small', 'Not Bird', 'Ramp Up and Can It', 'Stretch a Ramp', 'Max 2 Boxes']

def readInputFile(PathName, FileName):
if not os.path.exists(PathName):
    os.makedirs(PathName)

FilePath = PathName + "/" + FileName
total_data = pd.read_csv(FilePath)

return total_data


# Diccionario con los ids de usuario y número: uDict
def usersDict(pathName, fileName):
    ufile = pathName + "/" + fileName
    csv_file = csv.reader(open(ufile, "r"), delimiter= str(','))
    next(csv_file)
    mapUsers = {}
    mapGroups = {}
    cont =0
    for row in csv_file:
        user = row[student_column_number]
        group = row[group_column_number]
        if user not in mapUsers:
            mapUsers[user]=cont
            mapGroups[user] = group
            cont = cont+1
    return mapUsers, mapGroups


# Diccionario en el que el nombre del paso se asigna como un nombre de pregunta distinto al diccionario: qDict
def puzzlesDict(pathName, fileName):
    qfile = pathName + "/" + fileName
    csv_file = csv.reader(open(qfile, "r"), delimiter= str(','))
    next(csv_file)
    mapPuzzles = {}
    cont =0
    for row in csv_file:
        question = row[puzzle_column_number]
        if question not in mapPuzzles:
            mapPuzzles[question]=cont
            cont = cont+1
    return mapPuzzles



# Diccionario en el que el KC se mapea como etiquetas del diccionario: kcDict
def kcsDict(pathName, fileName):
    qfile = pathName + "/" + fileName
    QT = []
    csv_file = csv.reader(open(qfile, "r"), delimiter= str(','))
    next(csv_file)
    mapKc = {}
    cont =0
    for row in csv_file:
        tags = row[kc_column_number]
        if tags:
            tag = tags.split("~")
            for topics in tag:
                if topics not in mapKc:
                    mapKc[topics]=cont
                    cont = cont + 1
    return mapKc

def createKcDict(pathName, fileName):
    
    QTMat = dict()
    qfile = pathName + "/" + fileName
    csv_file = csv.reader(open(qfile, "r"), delimiter=",")
    next(csv_file)
    #cont=0
    for row in csv_file:
        qid = row[puzzle_column_number]
        kcs = row[kc_column_number]
        if(qid not in QTMat.keys()):
            #questions[cont]=qid
            QTMat[qid]=dict()
            #cont=cont+1
        if kcs:
            kc = kcs.split("~")
            for k in kc:
                #if qid in qDict and k in tDict:
                QTMat[qid][k] =0


    for puzzle in QTMat.keys():
        tam = len(QTMat[puzzle])
        #Se comprueba que tenga al menos un kc
        if tam>0:
            if(puzzle in mg1Puzzles):
                QTMat[puzzle]['MG.1'] = 0.5
                for x in QTMat[puzzle].keys():
                    if(x != 'MG.1'):
                        QTMat[puzzle][x] = 0.5/(tam-1)
            elif(puzzle in gmd4Puzzles):
                QTMat[puzzle]['GMD.4'] = 0.5
                for x in QTMat[puzzle].keys():
                    if(x != 'GMD.4'):
                        QTMat[puzzle][x] = 0.5/(tam-1)
            elif(puzzle in co5Puzzles):
                QTMat[puzzle]['CO.5'] = 0.5
                for x in QTMat[puzzle].keys():
                    if(x != 'CO.5'):
                        QTMat[puzzle][x] = 0.5/(tam-1)
            elif(puzzle in co6Puzzles):
                QTMat[puzzle]['CO.6'] = 0.5
                for x in QTMat[puzzle].keys():
                    if(x != 'CO.6'):
                        QTMat[puzzle][x] = 0.5/(tam-1)
            else:
                #Se dividen a partes iguales los kc
                for x in QTMat[puzzle].keys():
                    QTMat[puzzle][x] = 1/tam
    return QTMat



def loadDataset(inputfolder):
uDict, gDict = usersDict(inputfolder, datafile)
qDict =puzzlesDict(inputfolder, datafile)
kcDict =kcsDict(inputfolder, datafile)
kcsPuzzleDict =  createKcDict(inputfolder, datafile)

return uDict, gDict,qDict,kcDict, kcsPuzzleDict

#Obtener un valor RMSE basado en las predicciones del modelo y las respuestas reales
def rmseFunction(prob, ans, lenProb):
    prob = np.array(prob)
    ground = np.array(ans)
    error = (prob - ans)
    err_sqr = error*error
    rmse = math.sqrt(err_sqr.sum()/lenProb)
    return rmse




#Obtener un valor de accuracy basado en las predicciones de los modelos y las respuestas reales
def accuracyFunction(ans, prob):
    ans = np.array(ans)
    prob = np.array(prob)
    prob[prob >= 0.5] = 1
    prob[prob < 0.5] = 0
    acc = metrics.accuracy_score(ans, prob)
    return acc


def multiTopic_ELO(inputData, Competency, Diff,groupDiff, A_count, Q_count, kcsPuzzleDict ,gDict,gamma, beta):

alpha = 1
alpha_denominator = 0
correct = 0
prob_test = dict()
ans_test = dict()

response = np.zeros((len(inputData), 1))

for count, (index, item) in enumerate(inputData.iterrows()):
    alpha_denominator = 0
    uid = item[student_id]
    qid = item[puzzle_name]
    diff = Diff[qid]
    comp= dict()
    comp[uid]=[]
    for k in kcsPuzzleDict[qid]:
        comp[uid].append(Competency[uid][k] * kcsPuzzleDict[qid][k])
    compTotal = np.sum(comp[uid])
    probability = (1)/(1 + math.exp( -1 * (compTotal - diff)))
    if(uid not in prob_test.keys()):
        prob_test[uid] = dict()
    prob_test[uid][qid]=probability
    q_answered_count = Q_count[qid]
    
    if item[completed] == 1:

        response[count] = 1
        correct = 1
    else:
        response[count] = 0
        correct = 0
    
    #Se almacena la respuesta
    if(uid not in ans_test.keys()):
        ans_test[uid] = dict()
    ans_test[uid][qid] = correct
    
    groupDiff[gDict[uid]][qid] = groupDiff[gDict[uid]][qid] + ((gamma)/(1 + beta * q_answered_count)) * (probability - correct)
    
    Diff[qid] = Diff[qid] + ((gamma)/(1 + beta * q_answered_count)) * (probability - correct)
    Q_count[qid] += 1
    
    #Se calcula alpha
    alpha_numerator = probability - correct
    for k in kcsPuzzleDict[qid]:
        #if(T[qid][k] != 0):
        c_lambda = Competency[uid][k]
        probability_lambda = (1)/(1 + math.exp( -1 * (c_lambda - diff)))
        alpha_denominator = alpha_denominator + (correct - probability_lambda)
    alpha = abs(alpha_numerator / alpha_denominator)

    #Actualizando el nivel de competencia del estudiante en cada pregunta con la que la pregunta está etiquetada
    for k in kcsPuzzleDict[qid]:
        #if(T[qid][k] != 0):
        u_answered_count = A_count[uid][k]
        c = Competency[uid][k]
        probability = (1)/(1 + math.exp( -1 * (compTotal - diff)))
        
        Competency[uid][k] = Competency[uid][k]+kcsPuzzleDict[qid][k] * (gamma)/(1 + beta * u_answered_count) * alpha * (correct - probability)
        #Competency[uid][k] = Competency[uid][k]+ (gamma)/(1 + beta * u_answered_count) * alpha * (correct - probability)
        #print("Pregunta: ", qid)
        #print("Competency[uid][k]",uid,"-",k, Competency[uid][k])
        #print("kcsPuzzleDict[qid][k]",uid,"-",k, kcsPuzzleDict[qid][k])
        A_count[uid][k] += 1
            
return Competency, Diff,groupDiff, A_count , Q_count, prob_test, ans_test


def runExperiment(model, inputfolder, gamma, beta):
#Se cargan los datos y se rellenan las estructuras de datos
uDict,gDict,qDict,kcDict,kcsPuzzleDict = loadDataset(inputfolder)

#Se leen los archivos separados en train y test
train_set = readInputFile(inputfolder, trainfile)
test_set = readInputFile(inputfolder, testfile)

if model == 'multiTopic':
    
    group_difficulty = dict()
    question_difficulty = dict()
    question_counter = dict()
    for g in gDict.values():
        group_difficulty[g] = dict()
        for q in qDict.keys():
            question_difficulty[q]=0
            question_counter[q]=0
            group_difficulty[g][q]=0
    
    
    learner_competency = dict()
    response_counter = dict()
    for user in uDict.keys():
        if(user not in learner_competency.keys()):
            learner_competency[user]=dict()
            response_counter[user]=dict()
        for k in kcDict.keys():
            learner_competency[user][k]=0
            response_counter[user][k]=0


    learner_competency_train, question_difficulty_train,group_difficulty_train, response_counter_train, question_counter_train, prob_train, ans_train   = multiTopic_ELO(train_set, learner_competency, question_difficulty,group_difficulty, response_counter, question_counter, kcsPuzzleDict,gDict,gamma, beta)
    learner_competency_test, question_difficulty_test,group_difficulty_test, response_counter_test, question_counter_test, prob_test, ans_test   = multiTopic_ELO(test_set, learner_competency_train, question_difficulty_train,group_difficulty_train, response_counter_train, question_counter_train, kcsPuzzleDict,gDict,gamma, beta)


# tamaño 300
#print("Matriz: ")
#print(kcsPuzzleDict)
#print("Competencia de aprendizaje: ")
#print(learner_competency)

#print("Dificultad de cuestiones: ")
#print(question_difficulty)

#print("Contador de respuestas de cada alumno: ")
#print(response_counter)

#print("Contador de respuestas a cada pregunta: ")
#print(question_counter)

#print("prob_train: ", prob_train)
#print("p_test: ", p_test)
#print("a_test: ", a_test)

######### Normalización competency ###########
totalCompetencyGMD = []
totalCompetencyCO5 = []
totalCompetencyCO6 = []
totalCompetencyMG1 = []

for user in learner_competency.keys():
    for x in learner_competency[user]:
        if(x == 'GMD.4'):
            totalCompetencyGMD.append(learner_competency[user][x])
        elif(x == 'CO.5'):
            totalCompetencyCO5.append(learner_competency[user][x])
        elif(x == 'CO.6'):
            totalCompetencyCO6.append(learner_competency[user][x])
        elif(x == 'MG.1'):
            totalCompetencyMG1.append(learner_competency[user][x])
        
minCompetencyGMD = min(totalCompetencyGMD)
maxCompetencyGMD = max(totalCompetencyGMD)

minCompetencyCO5 = min(totalCompetencyCO5)
maxCompetencyCO5 = max(totalCompetencyCO5)

minCompetencyCO6 = min(totalCompetencyCO6)
maxCompetencyCO6 = max(totalCompetencyCO6)

minCompetencyMG1 = min(totalCompetencyMG1)
maxCompetencyMG1 = max(totalCompetencyMG1)

normalized_learner_competency = dict()
for user in learner_competency.keys():
    normalized_learner_competency[user]=dict()
    for x in learner_competency[user]:
        if(x == 'GMD.4'):
            normalized_learner_competency[user][x]= (learner_competency[user][x]- minCompetencyGMD)/(maxCompetencyGMD-minCompetencyGMD)
        elif(x == 'CO.5'):
            normalized_learner_competency[user][x]= (learner_competency[user][x]- minCompetencyCO5)/(maxCompetencyCO5-minCompetencyCO5)
        elif(x == 'CO.6'):
            normalized_learner_competency[user][x]= (learner_competency[user][x]- minCompetencyCO6)/(maxCompetencyCO6-minCompetencyCO6)
        elif(x == 'MG.1'):
            normalized_learner_competency[user][x]= (learner_competency[user][x]- minCompetencyMG1)/(maxCompetencyMG1-minCompetencyMG1)
        

#print(group_difficulty)
######## Normalización difficulty ###########

normalized_question_difficulty = dict()
for puzzle in question_difficulty.keys():
    if(puzzle not in question_difficulty.keys()):
        normalized_question_difficulty[puzzle] = 0
    normalized_question_difficulty[puzzle] = (question_difficulty[puzzle]-min(question_difficulty.values()))/(max(question_difficulty.values())-min(question_difficulty.values()))
    
######## Normalización group difficulty ###########
normalized_group_difficulty = dict()
for group in group_difficulty.keys():
    normalized_group_difficulty[group] = dict()
    for puzzle in group_difficulty[group].keys():
        normalized_group_difficulty[group][puzzle]=0
        normalized_group_difficulty[group][puzzle] = (group_difficulty[group][puzzle]-min(group_difficulty[group].values()))/(max(group_difficulty[group].values())-min(group_difficulty[group].values()))
   

#print(normalized_group_difficulty)
group_prob_test = []
for user in prob_test.keys():
    for task in prob_test[user].keys():
        group_prob_test.append(prob_test[user][task])
        
group_ans_test = []
for user in ans_test.keys():
    for task in ans_test[user].keys():
        group_ans_test.append(ans_test[user][task])
               
rmse = rmseFunction(group_prob_test, group_ans_test, len(group_prob_test))
accuracy = accuracyFunction(group_ans_test, group_prob_test)

return rmse, accuracy
#return normalized_group_difficulty



rmse_multiTopic, acc_multiTopic = runExperiment('multiTopic', inputfolder, 1.8, 0.05)



