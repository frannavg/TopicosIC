import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning) #ignorar warnings
import seaborn as sns #grafico
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1 - LIMPEZA COLUNA DATASETS
fifa = pd.read_csv('data.csv')

drop_cols = fifa.columns[28:54] #colunas com nivel do jogador em outras posicoes
fifa = fifa.drop(drop_cols, axis = 1) #drop delas
fifa = fifa.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',
               'Weight','Height','Contract Valid Until','Wage','Value','Name','Club'], axis = 1) #drop de algumas outras colunas nao interessantes
fifa = fifa.dropna()


# 2 - TRANSFORMACAO DE COLUNAS EM BINARIO

#variavel real face - y/n em binario
def face_to_num(fifa):
    if (fifa['Real Face'] == 'Yes'):
        return 1
    else:
        return 0

#pe preferido - string por binario
def right_footed(fifa):
    if (fifa['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

#posicao - criar posicoes mais genericas
def simple_position(fifa):
    if (fifa['Position'] == 'GK'):
        return 'GK'
    elif ((fifa['Position'] == 'RB') | (fifa['Position'] == 'LB') | (fifa['Position'] == 'CB') | (fifa['Position'] == 'LCB') | (fifa['Position'] == 'RCB') | (fifa['Position'] == 'RWB') | (fifa['Position'] == 'LWB') ):
        return 'DF'
    elif ((fifa['Position'] == 'LDM') | (fifa['Position'] == 'CDM') | (fifa['Position'] == 'RDM')):
        return 'DM'
    elif ((fifa['Position'] == 'LM') | (fifa['Position'] == 'LCM') | (fifa['Position'] == 'CM') | (fifa['Position'] == 'RCM') | (fifa['Position'] == 'RM')):
        return 'MF'
    elif ((fifa['Position'] == 'LAM') | (fifa['Position'] == 'CAM') | (fifa['Position'] == 'RAM') | (fifa['Position'] == 'LW') | (fifa['Position'] == 'RW')):
        return 'AM'
    elif ((fifa['Position'] == 'RS') | (fifa['Position'] == 'ST') | (fifa['Position'] == 'LS') | (fifa['Position'] == 'CF') | (fifa['Position'] == 'LF') | (fifa['Position'] == 'RF')):
        return 'ST'
    else:
        return fifa.Position


#nacionalidade - criacao de um major nation
nat_counts = fifa.Nationality.value_counts()
nat_list = nat_counts[nat_counts > 250].index.tolist()

#nacionalidade - reposicao para binario
def major_nation(fifa):
    if (fifa.Nationality in nat_list):
        return 1
    else:
        return 0


# 3 - NOVO DB E TARGET

#copia da database para evitar erros de indices
fifa1 = fifa.copy()

#aplicar mudancas, adicionando as novas colunas
fifa1['Real_Face'] = fifa1.apply(face_to_num, axis=1)
fifa1['Right_Foot'] = fifa1.apply(right_footed, axis=1)
fifa1['Simple_Position'] = fifa1.apply(simple_position,axis = 1)
fifa1['Major_Nation'] = fifa1.apply(major_nation,axis = 1)

#splitar work rate em 2
tempwork = fifa1["Work Rate"].str.split("/ ", n = 1, expand = True) 
#nova coluna para o primeiro workrate
fifa1["WorkRate1"]= tempwork[0]   
#nova coluna para o segundo workrate
fifa1["WorkRate2"]= tempwork[1]
#drop original das colunas utilizadas
fifa1 = fifa1.drop(['Work Rate','Preferred Foot', 'Real Face', 'Position', 'Nationality'], axis = 1)
fifa1.head()

#ID como valor alvo
target = fifa1.Overall
fifa2 = fifa1.drop(['Overall'], axis = 1)


# 4 - PREPARACAO E REGRESSAO

#teste, treinamento e categorizacao
X_train, X_test, y_train, y_test = train_test_split(fifa2, target, test_size=0.2)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
#print(X_test.shape,X_train.shape)
#print(y_test.shape,y_train.shape)

#aplicando a regressao
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#calculo do r2
print('r2 score: '+str(r2_score(y_test, predictions)))


# 5 - VISUALIZACAO

#Visualising the results
plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':0.7},line_kws={'color':'black','alpha':0.5})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Linear Prediction of Player Rating")
plt.show()