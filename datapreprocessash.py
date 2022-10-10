# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:05:49 2022

@author: HP
"""

import numpy as np
import pandas as pd

df = pd.read_excel(r"C:\Users\HP\Documents\Real estate valuation data set (1).xlsx", index_col=0)
df

##checking missing values##
null_col = df.columns[df.isnull().any()]
null_col
#thus from output we can say there are no missing values in data set
#here we can say that transcation date is not of much use 
#to see corelation between price and lat and longi
df.corr()
import matplotlib.pyplot as plt
plt.scatter(df["X1 transaction date"],df["Y house price of unit area"])
plt.show()
plt.scatter(df["X2 house age"],df["Y house price of unit area"])
plt.show()
plt.scatter(df["X3 distance to the nearest MRT station"],df["Y house price of unit area"])
plt.show()
plt.scatter(df["X4 number of convenience stores"],df["Y house price of unit area"])
plt.show()

var1= df["Y house price of unit area"]
var1
var2=df["X5 latitude"]
var2
np.corrcoef(var1, var2)
#this correlation is moderate
var3=df["X6 longitude"]
var3
np.corrcoef(var1,var3)
#this correlation is also moderate
df.info()
#now we can fetch this two var and can form one var name area for futher analysis 
df_cls = df.iloc[:,4:6]
df_cls
#to fetch lat and longi data

from sklearn.preprocessing import StandardScaler
#to create scaler
scaler = StandardScaler() 
df_cls_scaled = scaler.fit_transform(df_cls)
df_cls_scaled = pd.DataFrame(df_cls_scaled,columns=df_cls.columns,index=df_cls.index)
df_cls_scaled

#import kmeans
from sklearn.cluster import KMeans
#we have to find how many cluster will give best result
clustNos = [2,3,4,5,6,7,8,9,10]
inertia = []

for i in clustNos:
    model = KMeans(n_clusters=i,random_state=2019)
    model.fit(df_cls_scaled)
    inertia.append(model.inertia_)
print(i)

#import pyplot to visually decide the best nos of clusters
import matplotlib.pyplot as plt
plt.plot(clustNos, inertia, "-o")
plt.title("screeplot")
plt.xlabel('number of cluster, k')
plt.ylabel('inertia')
plt.xticks(clustNos)
plt.show()
#from here we see from from 3 point there is no much difference between points so we take 3 or 4 cluster

model.fit(df_cls_scaled)

# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=3, random_state=123, verbose=2)
model.fit(df_cls_scaled)
model.cluster_centers_

# Determine the cluster labels of new_points: labels
labels = model.predict(df_cls_scaled)
print(labels)
#we see each obs each allocted to one cluster i.e 0,1 or 2 

# Variation
print(model.inertia_)
#this value tells how well dataset was clustered using kmeans. 

clusterID = pd.DataFrame({'ClustID':labels})
clusterID['ClustID'].value_counts()
#cluserID gives how many datapoints does each cluster has
updated_df = pd.concat([df.reset_index(drop=True),clusterID],axis=1)

updated_df.info()
updated_df['ClustID'] = updated_df['ClustID'].astype(str)
updated_df.info()


#we have to drop unncessary var i.e transcation date and modify lat and longi var

updated_df = updated_df.drop(columns = ['X1 transaction date',
                                        'X5 latitude',
                                        'X6 longitude'])


updated_df = updated_df.rename(columns = {'ClustID': 'Area'})


updated_df['Area'] = updated_df['Area'].replace({'0': 'Area_1',
                                        '1': 'Area_2',
                                        '2': 'Area_3' })
                                        
    
updated_df.info()
updated_df

#now this data stored in updated_df can be used for further analysis
updated_df.to_csv(r"D:\Real estate valuation data set (1).csv", index = False)
updated_df
