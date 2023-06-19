#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#加载库处理数据
#加载数据
import pandas as pd 
import numpy as np
# 实现正态分布（高斯分布
from scipy.stats import norm
# 可视化
import seaborn as sns 

# 设置画图尺寸，坐标轴标题字体的大小
plt.rcParams['figure.figsize'] = (15,8) 
plt.rcParams['axes.titlesize'] = 'large'


# In[3]:


data = pd.read_csv('data/clean-data.csv', index_col=False)
data.drop('Unnamed: 0',axis=1, inplace=True)
data.head(2)


# In[3]:


#基础描述分析
data.describe()


# In[4]:


# 数据的偏度 查看数据的分布形态
data.skew()


#  >偏度结果显示正偏斜或负偏斜，数值的绝对值越大，表面数据分布越不对称，偏斜程度大。上图表，radius_mean, perimeter_mean, area_mean,concavity_mean 、concave_points_mean 在预测癌症类型方面是有用的

# In[10]:


# 以数组形式返回列的所有唯一值
data.diagnosis.unique()


# In[9]:


# 按‘diagnosis'诊断分组并输出查看
diag_gr = data.groupby('diagnosis', axis=0)
pd.DataFrame(diag_gr.size(), columns=['# of observations'])


# # 直方图数据可视化
# 
# 观察哪些特征对预测恶性或良性癌症最有帮助。查看可能有助于我们进行模型选择和超参数选择的总体趋势
# 

# In[6]:


#统一设置图片背景和尺寸
sns.set_style("white")
sns.set_context({"figure.figsize": (10, 8)})
# 直方图
sns.countplot(data['diagnosis'],label='Count',palette="Set3")


# In[ ]:


# 将特征分为三组：mean se worst
data_mean=data.iloc[:,2:12]
data_se=data.iloc[:,12:22]
data_worst=data.iloc[:,22:]

# print(data_mean.columns)
# print(data_se.columns)
# print(data_worst.columns)


# In[9]:


#各组特征可视化
hist_mean=data_mean.hist(bins=10, figsize=(15, 10),grid=False,)

#查看
#df_cut['radius_worst'].hist(bins=100)


# In[10]:


#查看分组后缀为_se的直方图
#hist_se=data_se.hist(bins=10, figsize=(15, 10),grid=False,)


# In[11]:


#查看分组后缀为_worst的直方图
#hist_worst=data_worst.hist(bins=10, figsize=(15, 10),grid=False,)


# # 密度图数据可视化

# In[12]:


#分组后缀_mean的密度图
plt = data_mean.plot(kind= 'density', subplots=True, layout=(4,3), sharex=False, 
                     sharey=False,fontsize=12, figsize=(15,10))


# In[13]:


#分组后缀为_se的密度图
#plt = data_se.plot(kind= 'density', subplots=True, layout=(4,3), sharex=False, 
#                     sharey=False,fontsize=12, figsize=(15,10))


# In[14]:


#分组后缀为_worst的密度图
#plt = data_worst.plot(kind= 'kde', subplots=True, layout=(4,3), sharex=False, sharey=False,fontsize=5, 
#                     figsize=(15,10))


# # 箱线图数据可视化

# ### Box plot "_mean" suffix designition

# In[15]:


# 分组后缀_mean的箱线图
#plt=data_mean.plot(kind= 'box' , subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=12)


# ### Box plot "_se" suffix designition

# In[16]:


# 分组后缀为_se的箱线图
#plt=data_se.plot(kind= 'box' , subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=12)


# ### Box plot "_worst" suffix designition

# In[17]:


#分组后缀为_worst的箱线图
#plt=data_worst.plot(kind= 'box' , subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=12)


# # 多模态数据可视化

# ### 相关矩阵

# In[18]:


# 画相关矩阵
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
sns.set_style("white")

data = pd.read_csv('data/clean-data.csv', index_col=False)
data.drop('Unnamed: 0',axis=1, inplace=True)
# 计算
corr = data_mean.corr()

# 遮罩
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 设置
data, ax = plt.subplots(figsize=(8, 8))
plt.title('Breast Cancer Feature Correlation')

# 自定义发散颜色
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# 使用蒙版和正确的纵横比绘制热图
sns.heatmap(corr, vmax=1.2, square='square', cmap=cmap, mask=mask, 
            ax=ax,annot=True, fmt='.2g',linewidths=2)


# In[19]:


# 散点图
plt.style.use('fivethirtyeight')
sns.set_style("white")

data = pd.read_csv('data/clean-data.csv', index_col=False)
g = sns.PairGrid(data[[data.columns[1],data.columns[2],data.columns[3],
                     data.columns[4], data.columns[5],data.columns[6]]],hue='diagnosis' )
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter, s = 3)

