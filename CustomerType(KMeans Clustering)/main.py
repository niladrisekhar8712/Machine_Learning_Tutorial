import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel('No of clusters')
plt.ylabel("WCSS")
plt.title('No of clusters vs WCSS')
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0, 1],c='red',s = 100, label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1, 1],c='blue',s = 100, label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0],X[y_kmeans==2, 1],c='green',s = 100, label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0],X[y_kmeans==3, 1],c='yellow',s = 100, label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0],X[y_kmeans==4, 1],c='magenta',s = 100, label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black', label='Cluster Centroid')
plt.legend()
plt.xlabel('Annual Income(in k$)')
plt.ylabel('Spending Score(1-100)')
plt.title('Income vs Spending Score')
plt.show()