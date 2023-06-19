#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


import pandas as pd 
import numpy as np
from scipy.stats import norm


import seaborn as sns 
plt.style.use('fivethirtyeight')
sns.set_style("white")


plt.rcParams['figure.figsize'] = (8,4) 


data = pd.read_csv('data/clean-data.csv', index_col=False)
data.drop('Unnamed: 0',axis=1, inplace=True)
#data.head()


# In[2]:


#对标签进行编码
# 将预测变量分配给 矩阵类型的变量
array = data.values
X = array[:,1:31]
y = array[:,0]


# In[3]:


#将类标签从其原始字符串表示形式（M 和 B）转换为整数
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#le.transform (['M', 'B'])


# > *After encoding the class labels(diagnosis) in an array y, the malignant tumors are now represented as class 1(i.e prescence of cancer cells) and the benign tumors are represented as class 0 (i.e no cancer cells detection), respectively*, illustrated by calling the transform method of LabelEncorder on two dummy variables.**
# 
# 
# #### Assesing Model Accuracy: Split data into training and test sets
# 
# The simplest method to evaluate the performance of a machine learning algorithm is to use different training and testing datasets. Here I will
# * Split the available data into a training set and a testing set. (70% training, 30% test)
# * Train the algorithm on the first part,
# * make predictions on the second part and 
# * evaluate the predictions against the expected results. 
# 
# The size of the split can depend on the size and specifics of your dataset, although it is common to use 67% of the data for training and the remaining 33% for testing.
# 

# ## 拆分训练集和测试集（7：3）

# In[4]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=7)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## 数据标准化

# In[5]:


from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
Xs = scaler.fit_transform(X)


# ##  PCA降维

# In[6]:


from sklearn.decomposition import PCA
# f特征提取
pca = PCA(n_components=10)
fit = pca.fit(Xs)


# In[7]:


# 取前两个PC 进行画图，查看降维后特征的区分度
X_pca = pca.transform(Xs)

PCA_df = pd.DataFrame()

PCA_df['PCA_1'] = X_pca[:,0]
PCA_df['PCA_2'] = X_pca[:,1]

plt.plot(PCA_df['PCA_1'][data.diagnosis == 'M'],PCA_df['PCA_2'][data.diagnosis == 'M'],'o', alpha = 0.7, color = 'r')
plt.plot(PCA_df['PCA_1'][data.diagnosis == 'B'],PCA_df['PCA_2'][data.diagnosis == 'B'],'o', alpha = 0.7, color = 'b')

plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(['Malignant','Benign'])
plt.show()


# ## 确定要保留的主成分数

# 为了确定应保留多少个主成分，通常通过制作碎石图来总结主成分分析的结果。

# In[9]:


var= pca.explained_variance_ratio_

#通过拐点确定选择前几个PC

plt.plot(var)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,shadow=False,markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

