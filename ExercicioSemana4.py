#iris dataset: Conjunto de dados consistem em 3 tipos diferentes de 
#petalas de iris (Setosa, Versicolour e Virginica) 
#e comprimento da sepala, armazenados em um numpy 150x4.
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy
from collections import Counter
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

""" Comparacao das petalas """

iris = datasets.load_iris()
X = iris.data[:,2:]  #Comparacao com duas ultimas caracteristicas
y = iris.target #classificacao
#0 Comprimento da sepala; 1 Largura da sepala; 
#2 comprimento da petala; 3 Largura da petala 
#setosa, versicolor, virginica
plt.subplots()
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Comprimento Petala')
plt.ylabel('Largura Petala')
plt.grid(True)
plt.show()


clf = MLPClassifier(alpha=0.01,max_iter=2000)
#dados de treinamento 'ate 40' de cada classe
yt=numpy.concatenate([y[:40], y[51:90], y[101:140]])
xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
clf.fit(xt, yt)  
#validacao com o restante dos dados
yv=numpy.concatenate([y[40:50], y[90:100], y[140:150]])
xv = numpy.concatenate([X[40:50,:], X[90:100,:], X[140:150,:]])
yp=clf.predict(xv)
print(yp)
print(yv)

comp = yp == yv
c= Counter(comp)
print(c)
#taxa de acerto
print(c[1]/(c[0]+c[1]))


titles_options = [("Matriz de confusao, sem normalizar", None),
                  ("Matriz de confusao normalizada", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, xv, yv,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


print(classification_report(yv, yp))



""" Comparacao dos comprimentos """

iris = datasets.load_iris()
X = iris.data[:,:3]  # as tres primeiras carateristicas (segunda descartada)
y = iris.target #classificacao
#0 Comprimento da sepala; 1 Largura da sepala; 
#2 comprimento da petala; 3 Largura da petala 
#setosa, versicolor, virginica
plt.subplots()
plt.scatter(X[:,0], X[:,2], c=y, cmap=plt.cm.Set1)
plt.xlabel('Comprimento Sepala')
plt.ylabel('Comprimento Petala')
plt.grid(True)
plt.show()


clf = MLPClassifier(alpha=0.01,max_iter=2000)
#dados de treinamento 'ate 40' de cada classe
yt=numpy.concatenate([y[:40], y[51:90], y[101:140]])
xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
clf.fit(xt, yt)  
#validacao com o restante dos dados
yv=numpy.concatenate([y[40:50], y[90:100], y[140:150]])
xv = numpy.concatenate([X[40:50,:], X[90:100,:], X[140:150,:]])
yp=clf.predict(xv)
print(yp)
print(yv)

comp = yp == yv
c= Counter(comp)
print(c)
#taxa de acerto
print(c[1]/(c[0]+c[1]))


titles_options = [("Matriz de confusao, sem normalizar", None),
                  ("Matriz de confusao normalizada", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, xv, yv,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


print(classification_report(yv, yp))



""" Comparacao das larguras """

iris = datasets.load_iris()
X = iris.data[:,1:4]  # as tres ultimas carateristicas (caracteristica #2 descartada)
y = iris.target #classificacao
#0 Comprimento da sepala; 1 Largura da sepala; 
#2 comprimento da petala; 3 Largura da petala 
#setosa, versicolor, virginica
plt.subplots()
plt.scatter(X[:,0], X[:,2], c=y, cmap=plt.cm.Set1)
plt.xlabel('Largura Sepala')
plt.ylabel('Largura Petala')
plt.grid(True)
plt.show()


clf = MLPClassifier(alpha=0.01,max_iter=2000)
#dados de treinamento 'ate 40' de cada classe
yt=numpy.concatenate([y[:40], y[51:90], y[101:140]])
xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
clf.fit(xt, yt)  
#validacao com o restante dos dados
yv=numpy.concatenate([y[40:50], y[90:100], y[140:150]])
xv = numpy.concatenate([X[40:50,:], X[90:100,:], X[140:150,:]])
yp=clf.predict(xv)
print(yp)
print(yv)

comp = yp == yv
c= Counter(comp)
print(c)
#taxa de acerto
print(c[1]/(c[0]+c[1]))


titles_options = [("Matriz de confusao, sem normalizar", None),
                  ("Matriz de confusao normalizada", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, xv, yv,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


print(classification_report(yv, yp))



""" Naive Bayes Gaussiano """

X, y = load_iris(return_X_y=True)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_treinamento, y_treinamento).predict(X_teste)
print("Quantidade de pontos não rotulados (dentre de %d pontos) : %d" % ( X_teste.shape[0], (y_teste != y_pred).sum()))