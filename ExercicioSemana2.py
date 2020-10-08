import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import linear_model
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()

#print (diabetes.DESCR)



tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela.head()

print (tabela.head(10))



tabela['y'] = diabetes.target
tabela.head(10)

print (tabela['y'])


#Caracteristicas escolhidas: sex e bp

print ('\nSEX\n')

#1a - Sex

X = tabela['sex']
X_treinamento = X[:-20].array.to_numpy().reshape(-1,1)
X_teste = X[-20:].array.to_numpy().reshape(-1,1)

y = tabela['y']
y_treinamento = y[:-20]
y_teste = y[-20:].array.to_numpy().reshape(-1,1)


regr = linear_model.LinearRegression()

regr.fit(X_treinamento, y_treinamento)

#coeficientes b1
print('Coeficiente b1: \n', regr.coef_)
#intercepto b0
print('Coeficiente b0: \n', regr.intercept_)


diabetes_y_pred = regr.predict(X_teste)

print('Erro medio: ', sum(abs(y_teste-diabetes_y_pred.reshape(-1,1))))

plt.scatter(X_teste,y_teste,  color='black')
plt.plot(X_teste, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#end 1a

print ('\nBP\n')

#1b - Bp

X = tabela['bp']
X_treinamento = X[:-20].array.to_numpy().reshape(-1,1)
X_teste = X[-20:].array.to_numpy().reshape(-1,1)

y = tabela['y']
y_treinamento = y[:-20]
y_teste = y[-20:].array.to_numpy().reshape(-1,1)


regr = linear_model.LinearRegression()

regr.fit(X_treinamento, y_treinamento)

#coeficientes b1
print('Coeficiente b1: \n', regr.coef_)
#intercepto b0
print('Coeficiente b0: \n', regr.intercept_)


diabetes_y_pred = regr.predict(X_teste)

print('Erro medio: ', sum(abs(y_teste-diabetes_y_pred.reshape(-1,1))))

plt.scatter(X_teste,y_teste,  color='black')
plt.plot(X_teste, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#end 1b

print ('\nCombinacao BP e Sex\n')

#2 - Combinacao

X = tabela['sex']
X_treinamento = X[:-20].array.to_numpy().reshape(-1,1)
X_teste = X[-20:].array.to_numpy().reshape(-1,1)

y = tabela['bp']
y_treinamento = y[:-20]
y_teste = y[-20:].array.to_numpy().reshape(-1,1)


regr = linear_model.LinearRegression()

regr.fit(X_treinamento, y_treinamento)

#coeficientes b1
print('Coeficiente b1: \n', regr.coef_)
#intercepto b0
print('Coeficiente b0: \n', regr.intercept_)


diabetes_y_pred = regr.predict(X_teste)

print('Erro medio: ', sum(abs(y_teste-diabetes_y_pred.reshape(-1,1))))

plt.scatter(X_teste,y_teste,  color='black')
plt.plot(X_teste, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()