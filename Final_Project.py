#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease
# 
# # INFO 1998: Final Project
# 
# # Net ID: mrr224

# Data comes from https://www.kaggle.com/ronitf/heart-disease-uci
# 
# Problem Statement: 
# 
# The presence of heart disease in Americans is an increasing concern in today's world. A surprising number of factors are correlated with having heart disease, and these factors are used to develop the following machine learning models. The question is: "Does this person have heart disease or not, based on their health numbers?" The following set of machine learning models attempt to answer this question.
# 
# The factors that are considered in the following ML models to predict heart disease are as follows, taken from the source of the data: 
# 
# age = age in years
# sex: 1 = male; 0 = female
# cp = chest pain type (4 values)
# trestbps = resting blood pressure
# chol = serum cholesterol in mg/dl
# fbs = fasting blood sugar > 120 mg/dl
# restecg = resting electrocardiographic results (values 0,1,2)
# thalach = maximum heart rate achieved
# exang = exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# slope = the slope of the peak exercise ST segment
# ca = number of major vessels (0-3) colored by flourosopy
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# Based on prior knowledge and intuitions, I hypothesize that age and serum cholesterol will contribute the most information to determining if a person has heart disease or not. I test my hypothesis using models seen below. 

# In[175]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import cdist


# # Pre-Processing and Statistical Analysis

# In[196]:


#Find which factors are most significant to this dataset

heart_PCA = heart.drop('target',axis=1)
heart_PCA = pd.DataFrame(preprocessing.scale(heart_PCA),columns=heart_PCA.columns)
heart_PCA.head()
pca = PCA().fit(heart_PCA)

# plot fraction of variance explained by each component

x = np.arange(13)
plt.bar(x, pca.explained_variance_ratio_)
plt.title('Fraction of Variance Explained by Each Component')
plt.xlabel('Component')
plt.ylabel('Fraction of Total Variance')
plt.show()


# In order to create ML models, it is appropriate to conduct a component analysis to determine which factors to include in the model. Based on the above graph, it is clear that component "0" contributes over 20% to the variation of the response variable. Thus, we examine the constituents of component 0 below to determine which predictors to try first when creating our models. 

# In[201]:


#Component 0 contributes the most to the variation of the output. Let's see what factor contributes most to component 0.

components = pd.DataFrame(pca.components_,columns=heart_PCA.columns)
components

#Feature oldpeak contributes most to the variation of component 0
#After oldpeak, thalach and exang contributes the most
#Let's try these and see what happens.


# I reject my hypothesis. It is clear that age and serum cholesterol are perhaps not the most significant predictors of heart disease. Instead, thalach and oldpeak contribute most significantly to the variation of the response variable. Now I investigate the correlation matrix to see which predictors are most closely correlated with the target. 

# In[254]:


heart.corr()


# Based on this correlation matrix, it is clear that oldpeak, exang, thalach, and cp are most closely correlated with whether a person has heart disease or not. Thus, we will consider these predictors before the others when creating our model. 

# # Supervised Learning Models 

# In[206]:


heart=pd.read_csv("heart.csv")
X = heart.drop(['target'], axis=1)
Y = heart['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
heart.head()


# In[331]:


#K-Nearest Neighbors Classifier

X_KNN = heart.drop(['target'], axis=1)
X_KNN = X_KNN.drop(['age'], axis=1)
X_KNN = X_KNN.drop(['sex'], axis=1)
X_KNN = X_KNN.drop(['trestbps'], axis=1)
X_KNN = X_KNN.drop(['chol'], axis=1)
X_KNN = X_KNN.drop(['fbs'], axis=1)
X_KNN = X_KNN.drop(['restecg'], axis=1)
X_KNN = X_KNN.drop(['slope'], axis=1)
X_KNN = X_KNN.drop(['ca'], axis=1)
X_KNN = X_KNN.drop(['thal'], axis=1)
Y_KNN = heart['target']
X_KNN_train, X_KNN_test, Y_KNN_train, Y_KNN_test = train_test_split(X_KNN, Y_KNN, test_size=0.2)
knn = KNeighborsClassifier()
knn.fit(X_KNN_train, Y_KNN_train)
knn_pred_train = knn.predict(X_KNN_train)
knn_pred_test = knn.predict(X_KNN_test)
print("KNN Train Accuracy: ", accuracy_score(Y_KNN_train, knn_pred_train))
print("KNN Test Accuracy: ", accuracy_score(Y_KNN_test, knn_pred_test))


# This KNN Classification model appears to decently model the trends of the data, though the test accuracy is often significantly below the training accuracy.

# In[360]:


#Decision Tree Classifier

# We drop several predictors to try to achieve a higher and more stable test accuracy

X2 = heart.drop(['target'], axis=1)
X2 = X2.drop(['age'],axis=1)
X2 = X2.drop(['fbs'],axis=1)
X2 = X2.drop(['slope'],axis=1)
X2 = X2.drop(['sex'],axis=1)
X2 = X2.drop(['ca'],axis=1)
X2 = X2.drop(['restecg'],axis=1)
X2 = X2.drop(['thal'],axis=1)
X2 = X2.drop(['chol'],axis=1)
X2 = X2.drop(['trestbps'],axis=1)
Y2 = heart['target']
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2)
model=tree.DecisionTreeClassifier(max_depth=5)
model.fit(X2_train,Y2_train)
dtree_pred_train = model.predict(X2_train)
dtree_pred_test = model.predict(X2_test) 
print("Decision Tree Train Accuracy: ", accuracy_score(dtree_pred_train, Y2_train))
print("Decision Tree Test Accuracy: ", accuracy_score(dtree_pred_test, Y2_test))


# This decision tree has been optimized with the max_depth parameter, such that its max_depth yields a test accuracy as high as possible. This model is more fitting than the previous KNN model; its test and training accuracies have increased by more than 10%. 

# In[374]:


#Binary Logistic Classifier
X_BL = heart.drop(['target'], axis=1)
X_BL = X_BL.drop(['age'], axis=1)
X_BL = X_BL.drop(['sex'], axis=1)
X_BL = X_BL.drop(['trestbps'], axis=1)
X_BL = X_BL.drop(['chol'], axis=1)
X_BL = X_BL.drop(['fbs'], axis=1)
X_BL = X_BL.drop(['restecg'], axis=1)
X_BL = X_BL.drop(['slope'], axis=1)
X_BL = X_BL.drop(['ca'], axis=1)
X_BL = X_BL.drop(['thal'], axis=1)
Y_BL = heart['target']
X_BL_train, X_BL_test, Y_BL_train, Y_BL_test = train_test_split(X_BL, Y_BL, test_size=0.2)
model2 = LogisticRegression()
model2.fit(X_BL_train,Y_BL_train)
binary_pred_train = model2.predict(X_BL_train)
binary_pred_test = model2.predict(X_BL_test) 
print("Binary Training Accuracy: ", accuracy_score(Y_BL_train,binary_pred_train))
print("Binary Test Accuracy: ", accuracy_score(Y_BL_test,binary_pred_test))


# This binary logistic classifier, while having a lower train accuracy, appears to generalize very well to new data; it has a mean test accuracy of nearly 80%. It has a singificantly better test accuracy than the previous two models. If I were to use any Supervised Learning Model to answer the question "do I have heart disease?", this would be the model. 

# # Unsupervised Learning Models

# In[195]:


#Hierarchial Clustering

heart2=heart.drop(['target'], axis=1)
heart2 = StandardScaler().fit_transform(heart2)
hclust = linkage(heart2)
dendrogram(hclust)
plt.show()


# In this Unsupervised ML Hierarchial Clustering model, it appears that the model was successful at extrapolating two target groups from the factors. However, it is largely unclear how correct this algorithm would be on future data sets, and whether or not it is even correctly classifying the results. Thus, we examine other methods of unsupervised learning. 

# In[75]:


heart3 = pd.DataFrame(X, columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
heart3_target = pd.DataFrame(Y, columns=['target'])

#heart3 = heart3.drop('age', axis=1)
heart3 = heart3.drop('oldpeak', axis=1)
heart3 = heart3.drop('sex', axis=1)
heart3 = heart3.drop('cp', axis=1)
heart3 = heart3.drop('fbs', axis=1)
heart3 = heart3.drop('restecg', axis=1)
#heart3 = heart3.drop('thalach', axis=1)
heart3 = heart3.drop('exang', axis=1)
heart3 = heart3.drop('trestbps', axis=1)
heart3 = heart3.drop('chol', axis=1)
heart3 = heart3.drop('slope', axis=1)
heart3 = heart3.drop('ca', axis=1)
heart3 = heart3.drop('thal', axis=1)
predictors_with_target = pd.concat([heart3, heart3_target], axis=1)
predictors_with_target.head()

k = len(predictors_with_target['target'].unique())
for i in predictors_with_target['target'].unique():
    # select only the applicable rows
    ds = predictors_with_target[predictors_with_target['target'] == i]
    # plot the points
    plt.plot(ds[['thalach']],ds[['age']],'o')
plt.title("Heart Disease")
plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Age (years)')
plt.show()

#Create KMeans Cluster Model

kmeans=cluster.KMeans(n_clusters=k)
kmeans.fit(heart3)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

_, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# original graph

for i in predictors_with_target['target'].unique():
    ds = predictors_with_target[predictors_with_target['target'] == i]
    # plot the points
    ax1.plot(ds[['thalach']],ds[['age']],'o')
ax1.set_title("Heart Disease Data")
ax1.set_xlabel('Maximum Heart Rate Achieved')
ax1.set_ylabel('Age (years)')

# kmeans graph

for i in range(k):
    ds = heart3.iloc[np.where(labels==i)]
    # plot the data observations
    ax2.plot(ds[['thalach']],ds[['age']],'o')
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
ax2.set_title("Heart Disease Predictions")
ax2.set_xlabel('Maximum Heart Rate Achieved')
ax2.set_ylabel('Age (years)')

plt.show()


# After several unsuccessful attempts of combining factors to produce meaningful output, I found that the most distinguishing features for a 2D analysis of this kind were thalach (max heart rate) and age. While the K-Means clustering in this example is not precise, it is the best that I was able to achieve given the dataset. On the other hand, I may pursue future use of this model by using three predictors instead of two, making it in 3D space. This would allow it to potentially communicate more meaningful information about whether a person has heart disease or not. As the model is right now, I would most likely not use it to determine if someone had heart disease. 

# # Conclusions and Recommendations for Further Study

# While problems naturally arise when using ML to predict outcomes, I attempted to resolve these issues throughout the exploratory process. I changed which factors I used depending on the correlation matrix and the PCA analysis. I started out with supervised learning models to classify someone as having heart disease or not. I used K-Nearest Neighbors, Binary Logistic Classification, and Decision Tree Classification. It is worth noting that all three methods achieved relatively high accuracy scores (above 50%) each trial, with a mild to moderate disparity between corresponding training and testing accuracy scores. If someone were interested in knowing if they had heart disease, I would advise using all three models in tandem to ensure a reliable answer to the question. If I were to continue modifying my algorithms, I would put greater emphasis on the sensitivity to ensure that those who in fact have heart disease are diagnosed as so. I would achieve this be reducing the probability of a type II error. 
# 
# During my time modeling the data with unsupervised ML algorithms, I found that there was a greater deal of complexity with analyzing how these algorithms performed. I used two unsupervised learning models: K-Means Clustering and Hierarchial Clustering. It was unclear if my K-Means Model would be of any use at all; it appeared to simply cut the data in half so as to classify it. The hierarchial clustering model seemed to have success finding two separate target responses. However, it was also unclear if it was accurately cluterting these outcomes or not. All in all, I would recommend the use of the supervised ML algorithms that I created, which have tangible accuracy scores, as opposed to the clustering analyses. 
