import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import normal_ad

diabetes = load_diabetes()


tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela['y'] = diabetes.target
y = tabela['y']
y_treinamento = y[:-20]
y_teste = y[-20:].array.to_numpy().reshape(-1,1)


""" BMI """

X_bmi = tabela['bmi']
X_treinamento_bmi = X_bmi[:-20].array.to_numpy().reshape(-1,1)
X_teste_bmi = X_bmi[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_bmi, y_treinamento)

diabetes_y_pred_bmi = regr.predict(X_teste_bmi)

res = y_teste - diabetes_y_pred_bmi.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de bmi =', r2_score(y_teste, diabetes_y_pred_bmi))

plt.scatter(X_teste_bmi, y_teste,  color='black')
plt.plot(X_teste_bmi, diabetes_y_pred_bmi, color='blue', linewidth=3)
plt.show()

""" BP """

X_bp = tabela['bp']
X_treinamento_bp = X_bp[:-20].array.to_numpy().reshape(-1,1)
X_teste_bp = X_bp[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_bp, y_treinamento)

diabetes_y_pred_bp = regr.predict(X_teste_bp)

res = y_teste - diabetes_y_pred_bp.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de bp =', r2_score(y_teste, diabetes_y_pred_bp))

plt.scatter(X_teste_bp, y_teste,  color='black')
plt.plot(X_teste_bp, diabetes_y_pred_bp, color='blue', linewidth=3)
plt.show()


""" SEX """

X_sex = tabela['sex']
X_treinamento_sex = X_sex[:-20].array.to_numpy().reshape(-1,1)
X_teste_sex = X_sex[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_sex, y_treinamento)

diabetes_y_pred_sex = regr.predict(X_teste_sex)

res = y_teste - diabetes_y_pred_sex.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de sex =', r2_score(y_teste, diabetes_y_pred_sex))

plt.scatter(X_teste_sex, y_teste,  color='black')
plt.plot(X_teste_sex, diabetes_y_pred_sex, color='blue', linewidth=3)
plt.show()


""" S1 """

X_s1 = tabela['s1']
X_treinamento_s1 = X_s1[:-20].array.to_numpy().reshape(-1,1)
X_teste_s1 = X_s1[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s1, y_treinamento)

diabetes_y_pred_s1 = regr.predict(X_teste_s1)

res = y_teste - diabetes_y_pred_s1.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s1 =', r2_score(y_teste, diabetes_y_pred_s1))

plt.scatter(X_teste_s1, y_teste,  color='black')
plt.plot(X_teste_s1, diabetes_y_pred_s1, color='blue', linewidth=3)
plt.show()


""" S2 """

X_s2 = tabela['s2']
X_treinamento_s2 = X_s2[:-20].array.to_numpy().reshape(-1,1)
X_teste_s2 = X_s2[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s2, y_treinamento)

diabetes_y_pred_s2 = regr.predict(X_teste_s2)

res = y_teste - diabetes_y_pred_s2.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s2 =', r2_score(y_teste, diabetes_y_pred_s2))

plt.scatter(X_teste_s2, y_teste,  color='black')
plt.plot(X_teste_s2, diabetes_y_pred_s2, color='blue', linewidth=3)
plt.show()


""" S3 """

X_s3 = tabela['s3']
X_treinamento_s3 = X_s3[:-20].array.to_numpy().reshape(-1,1)
X_teste_s3 = X_s3[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s3, y_treinamento)

diabetes_y_pred_s3 = regr.predict(X_teste_s3)

res = y_teste - diabetes_y_pred_s3.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s3 =', r2_score(y_teste, diabetes_y_pred_s3))

plt.scatter(X_teste_s3, y_teste,  color='black')
plt.plot(X_teste_s3, diabetes_y_pred_s3, color='blue', linewidth=3)
plt.show()


""" s4 """

X_s4 = tabela['s4']
X_treinamento_s4 = X_s1[:-20].array.to_numpy().reshape(-1,1)
X_teste_s4 = X_s4[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s4, y_treinamento)

diabetes_y_pred_s4 = regr.predict(X_teste_s4)

res = y_teste - diabetes_y_pred_s4.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s4 =', r2_score(y_teste, diabetes_y_pred_s4))

plt.scatter(X_teste_s4, y_teste,  color='black')
plt.plot(X_teste_s4, diabetes_y_pred_s4, color='blue', linewidth=3)
plt.show()


""" S5 """

X_s5 = tabela['s5']
X_treinamento_s5 = X_s5[:-20].array.to_numpy().reshape(-1,1)
X_teste_s5 = X_s5[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s5, y_treinamento)

diabetes_y_pred_s5 = regr.predict(X_teste_s5)

res = y_teste - diabetes_y_pred_s5.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s5 =', r2_score(y_teste, diabetes_y_pred_s5))

plt.scatter(X_teste_s5, y_teste,  color='black')
plt.plot(X_teste_s5, diabetes_y_pred_s5, color='blue', linewidth=3)
plt.show()



""" S6 """

X_s6 = tabela['s6']
X_treinamento_s6 = X_s6[:-20].array.to_numpy().reshape(-1,1)
X_teste_s6 = X_s6[-20:].array.to_numpy().reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s6, y_treinamento)

diabetes_y_pred_s6 = regr.predict(X_teste_s6)

res = y_teste - diabetes_y_pred_s6.reshape(-1,1)

plt.hist(res)
plt.title('Histograma dos residuos da regressao')
plt.show()

p_value = normal_ad(res)[1]
print('p-value abaixo 0.05 geralmente significa não normal:', round(p_value[0],2))
print()
if p_value < 0.05:
  print('Residuos não são normalmente distribuídos')
else:
  print('Residuos são normalmente distribuídos')

print ('R2 de s6 =', r2_score(y_teste, diabetes_y_pred_s6))

plt.scatter(X_teste_s6, y_teste,  color='black')
plt.plot(X_teste_s6, diabetes_y_pred_s6, color='blue', linewidth=3)
plt.show()

"""Analisando a tabela de correlacao foi possivel perceber que as variaveis mais correlatas (excluindo a propria diagonal principal) eh s1 com s2% (quase 90%)"""

regr = linear_model.LinearRegression()
regr.fit(X_treinamento_s1, X_treinamento_s2)

res1 = y_teste - diabetes_y_pred_s1.reshape(-1,1)
res2 = y_teste - diabetes_y_pred_s2.reshape(-1,1)

plt.hist(res1)
plt.title('Histograma dos residuos da regressao')
plt.show()

plt.scatter(X_teste_s1, y_teste,  color='black')
plt.scatter(X_teste_s2, y_teste,  color='green')
plt.plot(X_teste_s1, diabetes_y_pred_s1, color='blue', linewidth=3)
plt.plot(X_teste_s2, diabetes_y_pred_s2, color='red', linewidth=3)
plt.show()

