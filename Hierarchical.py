from sklearn.datasets import load_iris,load_wine
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score

iris = load_iris()
wine = load_wine()

X = iris.data
Y = iris.target

X_W = wine.data
Y_W = wine.target

#######Construcción del arbol para el dendograma utilizando la libreria

model_Hclust_ward = AgglomerativeClustering(affinity='euclidean',linkage='ward',distance_threshold=0,n_clusters=None)
model_Hclust_ward.fit(X)

model_Hclust_ward_w = AgglomerativeClustering(affinity='euclidean',linkage='ward',distance_threshold=0,n_clusters=None)
model_Hclust_ward_w.fit(X_W)

####Dendograma####

#print(model_Hclust_ward.labels_)
#print("")
#print(model_Hclust_ward.labels_.shape)
#print(model_Hclust_ward.children_)
#print("")
#print(model_Hclust_ward.distances_)

def plot_dendrogram(model, **kwargs):
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(model_Hclust_ward, color_threshold = 10)
plt.axhline(y=10,c='black',linestyle='--',label='altura de corte')
plt.legend(loc = 'best')
plt.show()

plot_dendrogram(model_Hclust_ward_w, color_threshold = 1000)
plt.axhline(y=1000,c='black',linestyle='--',label='altura de corte')
plt.legend(loc = 'best')
plt.show()

#####Ajuste del clustering Sabiendo el # de clusters a utilizar

new_model = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage = 'ward')
new_model.fit(X)

new_model_w = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage = 'ward')
new_model_w.fit(X_W)

pred = new_model.labels_
print(pred)
pred_W = new_model_w.labels_
print(pred_W)

fig = plt.figure('Datos agrupados')
ax = Axes3D(fig)
plt.plot(X[pred == 0,0],X[ pred == 0,2],X[pred == 0,3],'r.',label = 'Cluster 1')
plt.plot(X[pred == 1,0],X[ pred == 1,2],X[pred == 1,3],'b*',label = 'Cluster 2')
plt.plot(X[pred == 2,0],X[ pred == 2,2],X[pred == 2,3],'go',label = 'Cluster 3')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')
plt.legend(loc = 'best')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
plt.plot(X_W[pred_W == 0, 1],X_W[pred_W == 0,5],X_W[pred_W == 0,12],'r.', label='Cluster 1')
plt.plot(X_W[pred_W == 1, 1],X_W[pred_W == 1,5],X_W[pred_W == 1,12],'bx', label='Cluster 2')
plt.plot(X_W[pred_W == 2, 1],X_W[pred_W == 2,5],X_W[pred_W == 2,12],'g*', label='Cluster 3')
plt.plot(X_W[pred_W == 3, 1],X_W[pred_W == 3,5],X_W[pred_W == 3,12],'yo', label='Cluster 4')
ax.set_xlabel('Acido Malico')
ax.set_ylabel('Total Fenoles')
ax.set_zlabel('Proline')
plt.show()

score = f1_score(Y,pred, average ='macro')

print("F1_Score: ", 100*score, "%")
print(wine.DESCR)