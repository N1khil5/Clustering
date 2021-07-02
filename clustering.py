import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster as cluster

# Setting a random seed for the file to get the same results with every run.
np.random.seed(0)

# Dropping the channel and region column from the wholesale customers dataset
data = pd.read_csv("wholesale_customers.csv")
new_data = data.drop(labels=None, axis=1, columns='Channel')
new_data = new_data.drop(labels=None, axis=1, columns='Region')
'''
Q 2.1
# Printing the range and the mean for every column in the dataset.
'''
customersMin = np.min(new_data)
customersMax = np.max(new_data)
customersMean = np.mean(new_data)
print(new_data)
print()
print(customersMin)
print()
print(customersMax)
print()
print(customersMean)

'''
Q 2.2
Running k-means with k=3 and fitting to the data
References: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
'''

km = cluster.KMeans(n_clusters=3).fit(new_data)
kmPred = km.fit_predict(new_data)
new_data['cluster'] = kmPred

new_data1 = new_data[new_data.cluster == 0]
new_data2 = new_data[new_data.cluster == 1]
new_data3 = new_data[new_data.cluster == 2]

'''
Scatter plots for each pair of attributes.
References: https://stackoverflow.com/questions/28227340/kmeans-scatter-plot-plot-different-colors-per-cluster
'''
for i in new_data.columns:
    for j in new_data.columns:
        if i != j and i != 'cluster' and j != 'cluster':
            x = new_data.loc[:, i]
            y = new_data.loc[:, j]
            plt.scatter(x, y, c=kmPred, cmap=plt.cm.Paired)
            plt.xlabel(i)
            plt.ylabel(j)
            #plt.show()

'''
Q 2.3 
Running k-means for 3, 5 and 10 clusters. 

Between clusters was calculated using the sum of squared distances from cluster centers.
The between cluster distance was based on code from Data Mining Practical 4 solutions
Withing cluster distance was calculated using the inertia 

References:
1. https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.triu_indices.html
2. https://scikit-learn.org/stable/modules/clustering.html
3. Data Mining Practical 4 Solutions

'''
# Creating an array of zeros
between = np.zeros(3)
for i in range(3):
    between[i] = 0.0
    for l in range(i+1, 3):
        # Incrementing the between value with the sum of squared distances from cluster centers.
        between[i] += (np.square(km.cluster_centers_[i][0] - km.cluster_centers_[l][0]))
bc3 = np.sum(between)
print("bc3 = ", bc3)
''' 
Within cluster distance using loops in Practical 4 solutions result in the same result as finding the inertia so the
inertia code is utilised here.
'''
wcdist3 = km.inertia_
print("WC 3 = ", wcdist3)
score3 = bc3/wcdist3
print("score3=", score3)

# 5 clusters
km5 = cluster.KMeans(n_clusters=5).fit(new_data)
between5 = np.zeros(5)
for i in range(5):
    between5[i] = 0.0
    for l in range(i+1, 5):
        between5[i] += (np.square(km5.cluster_centers_[i][0] - km5.cluster_centers_[l][0]))
bc5 = np.sum(between5)
print("bc5 = ", bc5)
wcdist5 = km5.inertia_
print("WC 3 = ", wcdist5)
score5 = bc5/wcdist5
print("score5=", score5)

# 10 clusters
km10 = cluster.KMeans(n_clusters=10).fit(new_data)
between10 = np.zeros(10)
for i in range(10):
    between10[i] = 0.0
    for l in range(i+1, 10):
        between10[i] += (np.square(km10.cluster_centers_[i][0] - km10.cluster_centers_[l][0]))
bc10 = np.sum(between10)
print("bc10 = ", bc10)
wcdist10 = km10.inertia_
print("WC 10 = ", wcdist10)
score10 = bc10/wcdist10
print("score10=", score10)

