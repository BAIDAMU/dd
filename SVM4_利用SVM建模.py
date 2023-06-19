#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


import pandas as pd 
import numpy as np
from scipy.stats import norm

# 监督学习
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report

#可视化
import seaborn as sns 
plt.style.use('fivethirtyeight')
sns.set_style("white")

plt.rcParams['figure.figsize'] = (8,4) 


# In[2]:


data = pd.read_csv('data/clean-data.csv', index_col=False)


# In[4]:


array = data.values
X = array[:,1:31] 
y = array[:,0]
le = LabelEncoder()
y = le.fit_transform(y)

scaler =StandardScaler()
Xs = scaler.fit_transform(X)


# ## 交叉验证

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=2, stratify=y)

# 创建一个SVM分类器并拟合
clf = SVC(probability=True)
clf.fit(X_train, y_train)

#
classifier_score = clf.score(X_test, y_test)*100
print '\nThe classifier accuracy score is {:03.2f}\n'.format(classifier_score)


# In[14]:


#使用SVC估计器获得3倍交叉验证分数的平均值。
n_folds = 3
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))*100
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error)


# In[11]:


from sklearn.feature_selection import SelectKBest, f_regression
# 把特征选择与模型串联到一起组成一个pipline，便于进行模型的训练和预测
clf2 = make_pipeline(SelectKBest(f_regression, k=3),SVC(probability=True))
# 交叉验证（分类器，数据特征，数据标签，几折交叉检验）
scores = cross_val_score(clf2, Xs, y, cv=3)

n_folds = 3
# 该分类器的3倍交叉验证精度评分
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error)


# In[13]:


print scores
# 平均分
avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
print "Average score and uncertainty: (%.2f +- %.3f)%%"%avg


# ## 模型评估（ROC曲线）

# In[8]:


#混淆矩阵 即总结分类模型预测结果的情形分析表
# 每一列代表预测值，每一行代表实际类别
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
#print(cm)


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from IPython.display import Image, display

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j], 
                va='center', ha='center')
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values')
plt.show()
# 输出模型评估报告
print(classification_report(y_test, y_pred ))


# In[10]:


from sklearn.metrics import roc_curve, auc
#绘制ROC曲线
plt.figure(figsize=(10,8))
probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.axes().set_aspect(1)


# 为了正确地解释ROC，考虑沿对角线的点代表什么。在这些情况下，出现“+”和“-”的几率是相等的。因此，这与通过投掷无偏硬币进行预测并没有什么不同。简单地说，分类模型是随机的。
# 
# 对角线以上的点，tpr &gt;Fpr，模型说你处在一个比随机表现更好的区域。例如，假设tpr = 0.99, fpr = 0.01，则处于真阳性组的概率为(0.99/(0.99+0.01))=99%
# ． 此外，保持fpr不变，很容易看出，您的位置越垂直于对角线上方，分类模型就越好。
