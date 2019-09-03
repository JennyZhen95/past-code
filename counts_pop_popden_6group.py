from function import olsfunc
from function import olsfunc_multigroups
from function import broadgroup_plotting_scaling_distance_pop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import pylab as pl
import random as rd
import imageio
from numpy import array, matrix
from scipy.cluster.vq import vq, kmeans, whiten

"""
Separate counts of autism (2990) and add (3140)
"""
df = pd.read_csv('mental10_cleaned_new.csv') #5 million unique mental claims in 2010

autism = df[pd.to_numeric(df['DX1'].apply(lambda x: str(x)[:4] )) == 2990]
ad = df[pd.to_numeric(df['DX1'].apply(lambda x: str(x)[:4] )) == 3140]

autism.drop_duplicates(subset='ENROLID', inplace=True)
ad.drop_duplicates(subset='ENROLID', inplace=True)

autism_cty = pd.DataFrame()
autism_cty = autism_cty.append(autism['EMPCTY'].value_counts()).transpose()
autism_cty.index = np.int64(autism_cty.index)

ad_cty = pd.DataFrame()
ad_cty = ad_cty.append(ad['EMPCTY'].value_counts()).transpose()
ad_cty.index = np.int64(ad_cty.index)


"""
Rescaling
"""
MScounts = pd.read_csv('enroll10bycty.csv')
MScounts.drop(0,inplace=True)
MScounts['EMPCTY'] = MScounts['EMPCTY'].astype(np.int64)
MScounts.drop('PERCENT',axis=1,inplace=True)
MScounts.set_index('EMPCTY',inplace=True)

insured = pd.read_csv('countyinsured.csv',encoding='latin1')
insured = insured[['GEO.id2','HC01_EST_VC01','HC02_EST_VC01']]
insured.drop(index=0,inplace=True)
insured= insured.astype(int)
insured = insured.set_index ('GEO.id2')
insured['insured']=insured['HC01_EST_VC01']-insured['HC02_EST_VC01']


ad_cty['insuredbyMS'] = 0
ad_cty.loc[ad_cty.index.isin(MScounts.index),'insuredbyMS'] = MScounts['COUNT']

ad_cty['totalinsured'] = 0
ad_cty.loc[ad_cty.index.isin(insured.index),'totalinsured'] = insured['insured']

ad_cty.rename(columns={'EMPCTY': "raw"}, inplace=True)
ad_cty['rescaled'] = ad_cty['totalinsured']*ad_cty['raw']/ad_cty['insuredbyMS']
ad_cty = ad_cty[ad_cty['totalinsured'] != 0]

landcover = pd.read_csv('AAAcounts_landcover.csv')
landcover.drop(columns='counts_log', inplace=True)
ad_cty = pd.concat([ad_cty, landcover.set_index('Counties')], axis=1, join='inner')

ad_cty['rescaled_log'] = np.log(ad_cty['rescaled'])
ad_cty['raw_log'] = np.log(ad_cty['raw'])

"""
Check rescaling
"""
plt.clf()
sns.lmplot('raw_log','rescaled_log', data=ad_cty[ad_cty['raw_log']>3],scatter_kws={"s": 3})
ax = plt.gca()
ax.set_title('AD(H)D Rescaling')
olsfunc(ad_cty['raw_log'], ad_cty['rescaled_log'])
plt.show()



"""
pop_log vs counts_log
"""
plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==1], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group1')
olsfunc(main[main['group']==1]['pop_log'], main[main['group']==1]['counts_log'])
plt.show()


plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==2], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group2')
olsfunc(main[main['group']==2]['pop_log'], main[main['group']==2]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==3], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group3')
olsfunc(main[main['group']==3]['pop_log'], main[main['group']==3]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==4], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group4')
olsfunc(main[main['group']==4]['pop_log'], main[main['group']==4]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==5], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group5')
olsfunc(main[main['group']==5]['pop_log'], main[main['group']==5]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('pop_log', 'counts_log', data=main[main['group']==6], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group6')
olsfunc(main[main['group']==6]['pop_log'], main[main['group']==6]['counts_log'])
plt.show()


"""
popden_log vs counts_log

"""
plt.clf()
plt.scatter(ad_cty[ad_cty['raw_log']>3]['density_log'], ad_cty[ad_cty['raw_log']>3]['rescaled_log'], s=3)
plt.xlabel('density_log')
plt.ylabel('rescaled_log')
plt.show()

plt.clf()
plt.scatter(ad_cty[ad_cty['raw_log']>3]['rescaled_log'], ad_cty[ad_cty['raw_log']>3]['density_log'], s=3)
plt.xlabel('rescaled_log')
plt.ylabel('density_log')
plt.show()

plt.clf()
sns.lmplot('density_log', 'rescaled_log', data=ad_cty, scatter_kws={"s": 5})
plt.show()

plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==1], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group1')
olsfunc(main[main['group']==1]['density_log'], main[main['group']==1]['counts_log'])
plt.show()


plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==2], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group2')
olsfunc(main[main['group']==2]['density_log'], main[main['group']==2]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==3], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group3')
olsfunc(main[main['group']==3]['density_log'], main[main['group']==3]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==4], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group4')
olsfunc(main[main['group']==4]['density_log'], main[main['group']==4]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==5], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group5')
olsfunc(main[main['group']==5]['density_log'], main[main['group']==5]['counts_log'])
plt.show()

plt.clf()
sns.lmplot('density_log', 'counts_log', data=main[main['group']==6], scatter_kws={"s": 5})
ax = plt.gca()
ax.set_title('group6')
olsfunc(main[main['group']==6]['density_log'], main[main['group']==6]['counts_log'])
plt.show()



"""
K-mean Clustering:
popden vs aaa_counts
"""


# 计算平面两点的欧氏距离
step = 0
color = ['.r', '.g', '.b', '.y']  # 颜色种类
dcolor = ['*r', '*g', '*b', '*y']  # 颜色种类
frames = []


def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# K均值算法
def k_means(x, y, k_count):
    count = len(x)  # 点的个数
    # 随机选择K个点
    k = rd.sample(range(count), k_count)
    k_point = [[x[i], [y[i]]] for i in k]  # 保证有序
    k_point.sort()
    global frames
    global step
    while True:
        km = [[] for i in range(k_count)]  # 存储每个簇的索引
        # 遍历所有点
        for i in range(count):
            cp = [x[i], y[i]]  # 当前点
            # 计算cp点到所有质心的距离
            _sse = [distance(k_point[j], cp) for j in range(k_count)]
            # cp点到那个质心最近
            min_index = _sse.index(min(_sse))
            # 把cp点并入第i簇
            km[min_index].append(i)
        # 更换质心
        step += 1
        k_new = []
        for i in range(k_count):
            _x = sum([x[j] for j in km[i]]) / len(km[i])
            _y = sum([y[j] for j in km[i]]) / len(km[i])
            k_new.append([_x, _y])
        k_new.sort()  # 排序

        # 使用Matplotlab画图
        pl.figure()
        pl.title("N=%d,k=%d  iteration:%d" % (count, k_count, step))
        for j in range(k_count):
            pl.plot([x[i] for i in km[j]], [y[i] for i in km[j]], color[j % 4])
            pl.plot(k_point[j][0], k_point[j][1], dcolor[j % 4])
        pl.savefig("1.jpg")
        frames.append(imageio.imread('1.jpg'))
        if (k_new != k_point):  # 一直循环直到聚类中心没有变化
            k_point = k_new
        else:
            return km

step=0
color=['.r','.g','.b','.y']#颜色种类
dcolor=['*r','*g','*b','*y']#颜色种类
frames = []

x = np.array(main['counts_log'])
y = np.array(main['density_log'])
k_count = 6

km = k_means(x, y, k_count)
print
step
imageio.mimsave('k-means.gif', frames, 'GIF', duration=0.5)

# Scipy kmeans with 2 groups
features = main[['counts_log', 'density_log']]
whitened = whiten(np.array(features))

codebook, distortion = kmeans(whitened, 2)
cluster, _ = vq(whitened, codebook)

features['cluster'] = np.transpose(cluster)
cluster1 = features[features['cluster'] == 0][['counts_log', 'density_log']]
cluster2 = features[features['cluster'] == 1][['counts_log', 'density_log']]

plt.clf()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1.iloc[:,1], cluster1.iloc[:,0], s=5, color='r', label='Cluster 1')
ax.scatter(cluster2.iloc[:,1], cluster2.iloc[:,0], s=5, color='g', label='Cluster 2')
ax.legend()
ax.set_xlabel('density_log')
ax.set_ylabel('counts_log')
ax.set_title('kmean clustering on raw data')
plt.show()





# Scipy kmeans with 6 groups
features = np.array(main[['counts_log', 'density_log']])
whitened = whiten(features)
codebook, distortion = kmeans(whitened, 6)
cluster, _ = vq(whitened, codebook)

whitened = pd.DataFrame(whitened)
whitened.columns = ['counts_log', 'density_log']
whitened['cluster'] = np.transpose(cluster)
cluster1 = whitened[whitened['cluster'] == 0][['counts_log', 'density_log']]
cluster2 = whitened[whitened['cluster'] == 1][['counts_log', 'density_log']]
cluster3 = whitened[whitened['cluster'] == 2][['counts_log', 'density_log']]
cluster4 = whitened[whitened['cluster'] == 3][['counts_log', 'density_log']]
cluster5 = whitened[whitened['cluster'] == 4][['counts_log', 'density_log']]
cluster6 = whitened[whitened['cluster'] == 5][['counts_log', 'density_log']]

plt.clf()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1.iloc[:,1], cluster1.iloc[:,0], s=5, color='r', label='Cluster 1')
ax.scatter(cluster2.iloc[:,1], cluster2.iloc[:,0], s=5, color='g', label='Cluster 2')
ax.scatter(cluster3.iloc[:,1], cluster3.iloc[:,0], s=5, color='b', label='Cluster 3')
ax.scatter(cluster4.iloc[:,1], cluster4.iloc[:,0], s=5, color='k', label='Cluster 4')
ax.scatter(cluster5.iloc[:,1], cluster5.iloc[:,0], s=5, color='y', label='Cluster 5')
ax.scatter(cluster6.iloc[:,1], cluster6.iloc[:,0], s=5, color='c', label='Cluster 6')
ax.scatter(codebook[:, 1], codebook[:, 0], c='m', s=50, marker = '+')
ax.legend()
ax.set_xlabel('density_log')
ax.set_ylabel('counts_log')
ax.set_title('kmean clustering on normalized data')
plt.show()


features = main[['counts_log', 'density_log']]
features['cluster'] = np.transpose(cluster)
cluster1 = features[features['cluster'] == 0][['counts_log', 'density_log']]
cluster2 = features[features['cluster'] == 1][['counts_log', 'density_log']]
cluster3 = features[features['cluster'] == 2][['counts_log', 'density_log']]
cluster4 = features[features['cluster'] == 3][['counts_log', 'density_log']]
cluster5 = features[features['cluster'] == 4][['counts_log', 'density_log']]
cluster6 = features[features['cluster'] == 5][['counts_log', 'density_log']]

plt.clf()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1.iloc[:,1], cluster1.iloc[:,0], s=5, color='r', label='Cluster 1')
ax.scatter(cluster2.iloc[:,1], cluster2.iloc[:,0], s=5, color='g', label='Cluster 2')
ax.scatter(cluster3.iloc[:,1], cluster3.iloc[:,0], s=5, color='b', label='Cluster 3')
ax.scatter(cluster4.iloc[:,1], cluster4.iloc[:,0], s=5, color='k', label='Cluster 4')
ax.scatter(cluster5.iloc[:,1], cluster5.iloc[:,0], s=5, color='y', label='Cluster 5')
ax.scatter(cluster6.iloc[:,1], cluster6.iloc[:,0], s=5, color='c', label='Cluster 6')
ax.legend()
ax.set_xlabel('density_log')
ax.set_ylabel('counts_log')
ax.set_title('kmean clustering on raw data')
plt.show()


plt.clf()
sns.scatterplot('density_log', 'counts_log', data=main, hue='group')
plt.show()
