import enum
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

fifa = pd.read_csv('data.csv')

""" Dataset muito longo 18207 jogadores com 89 atributos cada.
Serão escolhidos 10 para levantamento de estatísticas base """

#Visão Geral
print (fifa)

#Colunas
for col in fifa.columns:
    print (col)

#Análise da média das colunas e histograma
#Idade

playersAges = fifa['Age']
sumAges = totalPlayers = 0
for i in playersAges:
    sumAges += playersAges[i]  
    totalPlayers+=1
mediaAges = sumAges/totalPlayers

print ('\nA media de idade dos jogadores eh de %.2f anos\n' % mediaAges)
plt.hist(playersAges)
plt.title('Histograma da idade dos jogadores')
plt.show()


#Salario

playersWages = fifa['Wage']
sumWages = 0
for i in playersWages:
    num = i
    if (len(num) > 2): #possibilidade de €0
        num = num[1:-1] #limpeza dos salarios
        sumWages += int(num)
mediaWages = sumWages/totalPlayers

print ('\nA media de salario dos jogadores eh de %.2f mil euros\n' % mediaWages)
plt.hist(playersWages)
plt.title('Histograma do salário dos jogadores')
plt.show()


#Pontuação geral

playersOverall = fifa['Overall']
sumOverall = 0
for i in playersOverall:
    sumOverall += playersOverall[i]  #devolve o numero incorreto (?)
mediaOverall = sumOverall/totalPlayers

print ('\nA media de pontuação dos jogadores eh de %.2f.\n' % mediaOverall)
plt.hist(playersOverall)
plt.title('Histograma da pontuação dos jogadores')
plt.show()

#Potencial

playersPotencial = fifa['Potential']
sumPotencial = 0
for i in playersPotencial:
    sumPotencial += playersPotencial[i] #devolve o numero incorreto (?)
mediaPotencial = sumPotencial/totalPlayers

print ('\nA media de potencial dos jogadores eh de %.2f.\n' % mediaPotencial)
plt.hist(playersPotencial)
plt.title('Histograma do potencial dos jogadores')
plt.show()

"""# Jogadores por posicao

playersPosition = fifa['Position']
class Posicao(enum.Enum):
    LS = 1
    ST = 2
    RS = 3
    LW = 4
    LF = 5
    CF = 6
    RF = 7
    RW = 8
    LAM = 9
    CAM = 10
    RAM = 11
    LM = 12
    LCM = 13
    CM = 14
    RCM = 15
    RM = 16
    LWB = 17
    LDM = 18
    CDM = 19
    RDM = 20
    RWB = 21
    LB = 22
    LCB = 23
    CB = 24
    RCB = 25
    RB = 26
    GK = 27

pos = []
for i in playersPosition:
    print(playersPosition[i])
    atual = playersPosition[i]"""

# Pe preferido

playersFoot = fifa['Preferred Foot']

left = len(fifa[playersFoot == 'Left'])
right = len(fifa[playersFoot == 'Right'])

print('Jogadores destros:' ,right)
print('Jogadores canhotos:' ,left)

