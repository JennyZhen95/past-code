from function import olsfunc
from function import olsfunc_multigroups
from function import broadgroup_plotting_scaling_distance_pop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


"""
Assign groups to each county
"""
ctygroup = pd.read_excel('NCHSURCodes2013.xlsx',encoding='latin1', index_col='FIPS code')

main = pd.read_csv('AAAcounts_landcover.csv', index_col='Counties')
main['group'] = np.nan
main.loc[main.index.isin(ctygroup.index), 'group'] = ctygroup['2013 code']
main.dropna(subset=['group'], inplace=True)  #1833 counties in total

main.to_csv('AAAcounts_landcover_group.csv')


"""
Grouping:
1 67counties
2 323counties
3 321counties
4 269counties
5 533counties
6 320counties

variables interested: water, devopen, devlow, devmed, devhigh, barren, grasslands, herbwetlands, density
"""


"""
Group the first three as group1, and last three as group0.
group1: 710 counties
group0: 1122 counties
"""
main['broadgroup'] = 0
main.loc[(main['group']==1) | (main['group']==2) | (main['group']==3), 'broadgroup'] = 1


# density_log
distance_data = broadgroup_plotting_scaling_distance_pop('density_log', main)
beta0_density, beta1_density = broadgroup_distance_analysis_pop_resampling('density_log', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_density,beta1_density)

# devlow
distance_data = broadgroup_plotting_scaling_distance_pop('devlow', main)
beta0_devlow, beta1_devlow = broadgroup_distance_analysis_pop_resampling('devlow', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_devlow,beta1_devlow)

# water
distance_data = broadgroup_plotting_scaling_distance_pop('water', main)
beta0, beta1 = broadgroup_distance_analysis_pop_resampling('water', distance_data)
cohen_d(beta0, beta1)

diff= beta0 - beta1
std_diff = np.sqrt(beta0.var()/beta0.shape[0] + beta1.var()/beta1.shape[0])
z_stat = diff.mean()/std_diff
import scipy
p_val = scipy.stats.norm.cdf(z_stat)
sns.distplot(diff)
plt.show()


# devopen
distance_data = broadgroup_plotting_scaling_distance_pop('devopen', main)
beta0_devopen, beta1_devopen = broadgroup_distance_analysis_pop_resampling('devopen', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_devopen, beta1_devopen)

#devmed
distance_data = broadgroup_plotting_scaling_distance_pop('devmed', main)
beta0_devmed, beta1_devmed = broadgroup_distance_analysis_pop_resampling('devmed', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_devmed, beta1_devmed)

#devhigh
distance_data = broadgroup_plotting_scaling_distance_pop('devhigh', main)
beta0_devhigh, beta1_devhigh = broadgroup_distance_analysis_pop_resampling('devhigh', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_devhigh, beta1_devhigh)


#grasslands
distance_data = broadgroup_plotting_scaling_distance_pop('grasslands', main)
beta0_grasslands, beta1_grasslands = broadgroup_distance_analysis_pop_resampling('grasslands', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_grasslands, beta1_grasslands)

#barren
distance_data = broadgroup_plotting_scaling_distance_pop('barren', main)
beta0_barren, beta1_barren = broadgroup_distance_analysis_pop_resampling('barren', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_barren, beta1_barren)

#herbwetlands
distance_data = broadgroup_plotting_scaling_distance_pop('herbwetlands', main)
beta0_herbwetlands, beta1_herbwetlands = broadgroup_distance_analysis_pop_resampling('herbwetlands', distance_data)
plot_multiple_distribution_with_cohen_d(beta0_herbwetlands, beta1_herbwetlands)


"""
Random sampling with replacement
"""
## Group1
group1 = main.loc[main.broadgroup == 1, ['counts_log','density_log','pop_log']]
group1.reset_index(col_fill='county', inplace=True)
replic1 = np.array([np.random.choice(group1.shape[0], 500, replace = True) for _ in range(1000)])

beta1 = np.zeros([1000, 1])
beta1_se = np.zeros([1000, 1])
for i in range(1000):
    df = group1.loc[replic1[i, :]]
    X = sm.add_constant(df['density_log'])
    results = sm.OLS(df['counts_log'], X).fit()
    beta1[i] = results.params[1]
    beta1_se[i] = results.bse[1]

sns.distplot(beta1, hist_kws=dict(edgecolor="k", linewidth=0.5), color='coral')
ax = plt.gca()
ax.text(
    0.01,
    0.8,
    r'$\bar \beta_1 = %0.3f$'
    % (beta1.mean()),
    transform=ax.transAxes, fontsize=10
)
plt.show()

## Group0
group0 = main.loc[main.broadgroup == 0, ['counts_log','density_log','pop_log']]
group0.reset_index(col_fill='county', inplace=True)
replic0 = np.array([np.random.choice(group0.shape[0], 500, replace = True) for _ in range(1000)])

beta0 = np.zeros([1000, 1])
beta0_se = np.zeros([1000, 1])
for i in range(1000):
    df = group0.loc[replic0[i, :]]
    X = sm.add_constant(df['density_log'])
    results = sm.OLS(df['counts_log'], X).fit()
    beta0[i] = results.params[1]
    beta0_se[i] = results.bse[1]

sns.distplot(beta0, hist_kws=dict(edgecolor="k", linewidth=0.5), color='b')
ax = plt.gca()
ax.text(
    0.01,
    0.8,
    r'$\bar \beta_0 = %0.3f$'
    % (beta0.mean()),
    transform=ax.transAxes, fontsize=10
)
plt.show()


## Visualize std error for linear coefficients
beta = pd.DataFrame([beta0.flatten(), beta0_se.flatten(), beta1.flatten(), beta1_se.flatten()]).transpose()
beta.columns = ['beta0', 'beta0_se', 'beta1', 'beta1_se']

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(beta.index[:101], beta.beta1[:101], color='coral', label='group1')
ax.plot(beta.index[:101], beta.beta0[:101], label='group0')
ax.fill_between(beta.index[:101], beta.beta1[:101] - beta.beta1_se[:101],
               beta.beta1[:101] + beta.beta1_se[:101],
                color='coral', alpha=0.1)
ax.fill_between(beta.index[:101], beta.beta0[:101] - beta.beta0_se[:101],
                beta.beta0[:101] + beta.beta0_se[:101],
                alpha=0.1)
ax.axhline(y=0.670, color='coral', linestyle='--')
ax.axhline(y=0.541, linestyle='--')
ax.legend(loc='best')
ax.set_xlabel('sample')
ax.set_ylabel(r'$\bar \beta$')
ax.set_title('First 100 samples')


plt.show()
plt.clf()



















# Deprecated
temp_group1 = temp[temp['group'] == 1]
p = sns.regplot('D_counts', 'D_variable', data=temp_group1)
olsfunc(temp_group1.D_counts, temp_group1.D_variable)
plt.show()

temp_group2 = temp[temp['group'] == 2]
p = sns.regplot('D_counts', 'D_variable', data=temp_group2)
olsfunc(temp_group2.D_counts, temp_group2.D_variable)
plt.show()

temp_group3 = temp[temp['group'] == 3]
p = sns.lmplot('D_counts', 'D_variable', data=temp_group3)
olsfunc(temp_group3.D_counts, temp_group3.D_variable)
plt.show()

temp_group4 = temp[temp['group'] == 4]
p = sns.lmplot('D_counts', 'D_variable', data=temp_group4)
olsfunc(temp_group4.D_counts, temp_group4.D_variable)
plt.show()

temp_group5 = temp[temp['group'] == 5]
p = sns.lmplot('D_counts', 'D_variable', data=temp_group5)
olsfunc(temp_group5.D_counts, temp_group5.D_variable)
plt.show()

temp_group6 = temp[temp['group'] == 6]
p = sns.lmplot('D_counts', 'D_variable', data=temp_group6)
olsfunc(temp_group6.D_counts, temp_group6.D_variable)
plt.show()







d["p{0}".format(i)].set_axis_labels("Distance_%s"%landcover.columns[i],
                                        'Distance_AAAcounts',
                                        fontsize=15) # Change xlabel name within loop
d["p{0}".format(i)].annotate(stats.pearsonr)
plt.tight_layout()





g = sns.pairplot(main, x_vars=['pop_log'], y_vars=main.columns[:17],
                 kind='reg', hue='group',
                 markers=".", size=5)
plt.gca().set_xlim((9,18))
plt.rcParams["axes.labelsize"] = 13
g.map(olsfunc)
plt.savefig('pairplot_PopLandcover4.png')


sns.lmplot(x='pop_log', y='density_log', hue='group', data=main,
           markers=".")

sns.lmplot(x='pop_log', y='counts_log', hue='group', data=main,
           markers=".")




"""
Visualize AAAcounts with other factors
"""
p = sns.pairplot(main, x_vars=main.columns[:17],
                 y_vars=['counts_log'], kind='reg', hue='group', markers='.')
p.map(olsfunc)
plt.savefig('pairplot_CountsLandcover.png')


"""
Distance Scaling
"""
d={}
for i in range(16):
    i=1
    temp = main.iloc[:, [i, 16, 17, 18]]
    temp.dropna(inplace=True)

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp['counts_log'],X).fit()
    temp['D_counts'] = results.resid

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp.iloc[:,0],X).fit()
    temp['D_variable'] = results.resid

    d["p{0}".format(i)] = sns.scatterplot(x=temp.D_variable, y=temp.D_counts,
                                         hue=temp.group)
    plt.xlabel("Distance_%s" % main.columns[i], fontsize=15)
    plt.ylabel('Distance_AAAcounts') # Change xlabel name within loop
    d["p{0}".format(i)].annotate(stats.pearsonr)
    plt.tight_layout()

locals().update(d) # Free variables from a dictionary


p0.savefig('p0.png')
p1.savefig('p1.png')
p2.savefig('p2.png')
p3.savefig('p3.png')
p4.savefig('p4.png')
p5.savefig('p5.png')
p6.savefig('p6.png')
p7.savefig('p7.png')
p8.savefig('p8.png')
p9.savefig('p9.png')
p10.savefig('p10.png')
p11.savefig('p11.png')
p12.savefig('p12.png')
p13.savefig('p13.png')
p14.savefig('p14.png')
p15.savefig('p15.png')

i=15
d={}
temp = main.iloc[:, [i, 16, 17, 18]]
temp.dropna(inplace=True)

X = sm.add_constant(temp['pop_log'])
results = sm.OLS(temp['counts_log'],X).fit()
temp['D_counts'] = results.resid

X = sm.add_constant(temp['pop_log'])
results = sm.OLS(temp.iloc[:,0],X).fit()
temp['D_variable'] = results.resid

d["p{0}".format(i)] = sns.scatterplot(x=temp.D_variable, y=temp.D_counts,
                                     hue=temp.group)
plt.xlabel("Distance_%s" % main.columns[i], fontsize=15)
plt.ylabel('Distance_AAAcounts') # Change xlabel name within loop
