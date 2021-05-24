import sklearn.datasets as sk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score

iris = sk.load_iris()
wine = sk.load_wine()

#print(iris.DESCR)
##############################
#### Asignación de Datos ##### 

X = iris.data
Y = iris.target
X_W = wine.data
Y_W = wine.target

#print("Datos de entrada: ")
#print(X)
#print("Target: ")
#print(Y)

###############################
## Metodo para saber cuantos CLusters (El codo)

range_n_cluster = range(1,10)
inertias = []

for n_cluster in range_n_cluster:
    model_kmeans = KMeans(
        n_clusters=n_cluster,
        init ='k-means++'
    )
    model_kmeans.fit(X)
    inertias.append(model_kmeans.inertia_)

plt.figure()
plt.plot(range_n_cluster,inertias,marker = 'o')
plt.show()

range_n_cluster_w = range(1,15)
inertias_w = []

for n_cluster_w in range_n_cluster_w:
    model_kmeans_w = KMeans(
        n_clusters=n_cluster_w,
        init ='k-means++'
    )
    model_kmeans_w.fit(X_W)
    inertias_w.append(model_kmeans_w.inertia_)

plt.figure()
plt.plot(range_n_cluster_w,inertias_w,marker = 'o')
plt.show()

###############################################
###### Entrenamiento del KMean ################

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

centroides  = kmeans.cluster_centers_
pred = kmeans.labels_

print("Cenroides: ")
print(centroides)
print("       ")
print("Predicción: ")
print(pred)

kmeans_w = KMeans(n_clusters=4, init='k-means++')
kmeans_w.fit(X_W)

centroides_W  = kmeans_w.cluster_centers_
pred_W = kmeans_w.labels_

print("Cenroides: ")
print(centroides_W)
print("       ")
print("Predicción: ")
print(pred_W)

##################################
####### Plotear los datos ########
plt.figure()
plt.plot(X[:,0],'r.',label = 'Sepal Length')
plt.plot(X[:,1],'b.',label = 'Sepal Width')
plt.plot(X[:,2],'y.',label = 'Petal Length')
plt.plot(X[:,3],'g.',label = 'Petal Width')
plt.legend(loc = 'best')
plt.show()

fig = plt.figure()
Axes3D(fig)
plt.plot(X[:,0],X[:,2],X[:,3],'k*', label = 'Datos de Entrada')
plt.show()

fig = plt.figure()
Axes3D(fig)
plt.plot(X_W[:,1],X_W[:,5],X_W[:,12],'k*', label = 'Datos de Entrada Wine')
plt.show()

#######################################################
### Plotear los datos con los cluster encontrados #####

fig = plt.figure()
ax = Axes3D(fig)
plt.plot(X[pred == 0, 0],X[pred == 0,2],X[pred == 0,3],'r.', label='Cluster 1')
plt.plot(X[pred == 1, 0],X[pred == 1,2],X[pred == 1,3],'bx', label='Cluster 2')
plt.plot(X[pred == 2, 0],X[pred == 2,2],X[pred == 2,3],'g*', label='Cluster 3')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')

plt.plot(centroides[:,0],centroides[:,2],centroides[:,3],'k*',markersize = 8, label = 'centroides')
plt.legend(loc = 'best')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
plt.plot(X_W[pred_W == 0, 1],X_W[pred_W == 0,5],X_W[pred_W == 0,12],'r.', label='Cluster 1')
plt.plot(X_W[pred_W == 1, 1],X_W[pred_W == 1,5],X_W[pred_W == 1,12],'bx', label='Cluster 2')
plt.plot(X_W[pred_W == 2, 1],X_W[pred_W == 2,5],X_W[pred_W == 2,12],'g*', label='Cluster 3')
plt.plot(X_W[pred_W == 3, 1],X_W[pred_W == 3,5],X_W[pred_W == 3,12],'yo', label='Cluster 4')
ax.set_xlabel('Alcohol')
ax.set_ylabel('')
ax.set_zlabel('')

plt.plot(centroides_W[:,1],centroides_W[:,5],centroides_W[:,12],'k*',markersize = 8, label = 'centroides')
plt.legend(loc = 'best')
plt.show()

################################################################
##### Comparación entre el target y la agrupación del KMean ####

f1 = f1_score(Y, pred,  average = 'macro')
print("F1 score: ", 100*f1, "%")



