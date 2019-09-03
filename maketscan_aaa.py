import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# df = pd.read_csv('/Volumes/Untitled/myfolders/mental10_cleaned_new.csv')
df = pd.read_csv('mental10_cleaned_new.csv') #5 million unique mental claims in 2010
df.info()
# Extract Autism, ADD and ADHD claims
"""
Autism: dx1 starts with 2990
ADD, ADHD: 3140
"""
autism = df[pd.to_numeric(df['DX1'].apply(lambda x: str(x)[:4] )) == 2990]
ad = df[pd.to_numeric(df['DX1'].apply(lambda x: str(x)[:4] )) == 3140]
df = pd.concat([autism,ad],ignore_index=True)

"""
Insured data for all counties
"""
insured = pd.read_csv('/Volumes/Untitled/myfolders/countyinsured.csv',encoding='latin1')
insured = insured[['GEO.id2','HC01_EST_VC01','HC02_EST_VC01']]
insured.drop(index=0,inplace=True)
insured= insured.astype(int)
insured = insured.set_index ('GEO.id2')
insured['insured']=insured['HC01_EST_VC01']-insured['HC02_EST_VC01']






"""
Rescaling
"""

AAAcounts.head(3)

AAAcounts = pd.DataFrame(df['EMPCTY'].value_counts())
AAAcounts.columns = ['counts']
AAAcounts.index.name='county'
AAAcounts.index = AAAcounts.index.astype(np.int64) # need to change index from float to integers

MScounts = pd.read_csv('/Volumes/Untitled/myfolders/enroll10bycty.csv')
MScounts.drop(0,inplace=True)
MScounts['EMPCTY'] = MScounts['EMPCTY'].astype(np.int64)
MScounts.drop('PERCENT',axis=1,inplace=True)
MScounts.set_index('EMPCTY',inplace=True)


AAAcounts['insuredbyMS'] = 0
AAAcounts.loc[AAAcounts.index.isin(MScounts.index),'insuredbyMS'] = MScounts['COUNT']

AAAcounts['totalinsured'] = 0
AAAcounts.loc[AAAcounts.index.isin(insured.index),'totalinsured'] = insured['insured']

AAAcounts['RescaledCounts'] = AAAcounts['totalinsured']*AAAcounts['counts']/AAAcounts['insuredbyMS']







AAAcounts = pd.read_csv('AAAcounts.csv')
AAAcounts.head(2)
"""
Visualize rescaling
"""
log = AAAcounts[AAAcounts.RescaledCounts != 0]
log_cleaned = pd.DataFrame(np.log(log['counts']))
log_cleaned['rescaled'] = np.log(log['RescaledCounts'])
#log.fillna(0,inplace=True)

log_cleaned.index = AAAcounts[AAAcounts.RescaledCounts != 0]['county']


# Calcualte rsquared, params and conf_int


X = sm.add_constant(log_cleaned['counts'])
results = sm.OLS(log_cleaned['rescaled'],X).fit()
results.rsquared
results.params
results.conf_int(alpha=0.05, cols=None)

# Plot rescaling with residual plots labelled
plt.clf()
sns.lmplot('counts','rescaled',data=log_cleaned, markers='.', scatter_kws={"s": 5})
plt.text(
    0.05,
    11.5,
    '$\\beta=%0.3f$ $[%0.3f,%0.3f]$ $r^2=%0.3f$'
    %(results.params[1],results.conf_int(alpha=0.05, cols=None).iloc[1,0],
    results.conf_int(alpha=0.05, cols=None).iloc[1,1],results.rsquared),
    fontsize=14,
    bbox=dict(boxstyle="round", alpha=0.5)
)
plt.show()
plt.savefig('rescaling.png')

resid_fitted = pd.concat([results.resid, results.fittedvalues], axis=1)
resid_fitted.columns = ['resid','fittedvalues']
resid_fitted.index = AAAcounts[AAAcounts.RescaledCounts != 0]['county']


fig,ax = plt.subplots()
ax.scatter('fittedvalues', 'resid', data=resid_fitted, s=0.7)
ax.set_xlabel('fitted values')
ax.set_ylabel('residual')
ax.axhline(y=0, color='b', linestyle='--', alpha=0.5)
ax.axhline(y=1.5, color='b', linestyle='--', alpha=0.5)
ax.axhline(y=-1.5, color='b', linestyle='--', alpha=0.5)

plt.savefig('rescaleResid.png')
for i, txt in enumerate(resid_fitted.index):
    ax.annotate(txt, (resid_fitted.fittedvalues.iloc[i], resid_fitted.resid.iloc[i]))

plt.savefig('RescaleResidual-label.png')







"""
Population density and population
"""
popden = pd.read_csv('2010populationdensity.csv',encoding='latin1')
popden = popden[~popden['GCT_STUB.display-label'].str.contains('Census Tract')]
popden = popden[['GCT_STUB.target-geo-id2','SUBHD0401','HD01']]
popden.drop(0,inplace=True)
sep = '('
popden['HD01'] = popden['HD01'].str.split(pat=sep,expand=True)[0] # Get the number before revision number
popden['GCT_STUB.target-geo-id2']=popden['GCT_STUB.target-geo-id2'].astype(np.int64) # Get rid of the zero before numbers
popden['HD01']=popden['HD01'].astype(np.int64)
popden = popden.set_index('GCT_STUB.target-geo-id2')
popden['county'] = popden.index
popden['SUBHD0401'] = popden['SUBHD0401'].astype(float)
popden['HD01'] = popden['HD01'].astype(float)
popden['SUBHD0401'].max()
popden['SUBHD0401'].min()



# Find the second largest population density number
max(n for n in popden['SUBHD0401'] if n!=max(popden['SUBHD0401']))

popden['SUBHD0401'].describe()

# Graph the population density on county map
colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
endpts = list(np.linspace(17, 129, len(colorscale) - 1))


values = popden['SUBHD0401'].tolist()
fips = popden['county'].tolist()


popden_fig = ff.create_choropleth(
    fips = fips, values = values, scope = ['usa'],
    binning_endpoints = endpts, colorscale = colorscale,
    show_state_data = False,
    show_hover = True, centroid_marker = {
        'opacity': 0
    },
    asp = 2.9,
    title = '2010 population density in USA',
    legend_title = 'density'
)
py.plot(popden_fig, filename = '2010 population density in USA')



"""
Visualize population density and AAAcounts in each county:
"""
counts_landcover = pd.read_csv('AAAcounts_landcover.csv')
counts_landcover.set_index('Counties',inplace=True)

#AAAcounts['totalpop'] = 0
#AAAcounts.loc[AAAcounts.index.isin(insured.index),'totalpop'] = insured['HC01_EST_VC01']
#AAAcounts['AAAperc'] = AAAcounts['RescaledCounts']/AAAcounts['totalpop']*100

#AAAcounts.to_csv('AAAcounts.csv')
#pop = pd.read_csv('countyinsured.csv',encoding='latin1')
#pop = pop[['GEO.id2','HD01_VD01']]
#pop.drop(0,inplace=True)

#pop['GEO.id2'] = pop['GEO.id2'].astype(np.int64) # change dtype of county id to integers in order to get rid of zeros
#pop['HD01_VD01'] = pop['HD01_VD01'].astype(int)
#pop.set_index('GEO.id2',inplace=True)
#pop.info()


counts_landcover['counts'] = np.exp(counts_landcover['counts_log'])
counts_landcover['density'] = np.exp(counts_landcover['density_log'])
counts_landcover['pop'] = np.exp(counts_landcover['pop_log'])

counts_landcover['count%'] = counts_landcover['counts']/counts_landcover['pop']*100

sns.distplot(counts_landcover['count%'])
plt.show()
pd.qcut(counts_landcover['count%'],5)

import plotly.plotly as py
import plotly.figure_factory as ff
import plotly
plotly.tools.set_credentials_file(username='JennyZhen95', api_key='Of7E528OP9ydcARd8vl7')


values = counts_landcover['count%'][counts_landcover['count%'].notnull()].tolist()  #0.147 - 4.889
fips = counts_landcover.index[counts_landcover['count%'].notnull()].tolist()

colorscale = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
endpts = [0.74, 0.985, 1.225, 1.534, 4.889]

fig = ff.create_choropleth(
    fips = fips, values = values, scope = ['usa'],
    binning_endpoints = endpts, colorscale = colorscale,
    county_outline={'color': 'rgb(15, 15, 55)', 'width': 0.3},
    show_state_data = True,
    show_hover = True, centroid_marker = {
        'opacity': 0
    },
    asp = 2.9,
    title = 'USA by AAA counts%',
    legend_title = '% AAA'
)
plotly.offline.plot(fig, filename = 'choropleth_full_usa')
py.plot(fig, filename = 'choropleth_full_usa')

## population density
values = counts_landcover['density'][counts_landcover['density'].notnull()].tolist()  #0.147 - 4.889
fips = counts_landcover.index[counts_landcover['density'].notnull()].tolist()

colorscale = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
endpts = [43.0, 70.6, 122.0, 308.78, 69468.4]

fig = ff.create_choropleth(
    fips = fips, values = values, scope = ['usa'],
    binning_endpoints = endpts, colorscale = colorscale,
    county_outline={'color': 'rgb(15, 15, 55)', 'width': 0.3},
    show_state_data = True,
    show_hover = True, centroid_marker = {
        'opacity': 0
    },
    asp = 2.9,
    title = 'USA by population density',
    legend_title = 'density'
)
plotly.offline.plot(fig, filename = 'choropleth_full_usa')
py.plot(fig, filename = 'choropleth_full_usa')
















"""
Visualize bivariate plot with AAAcounts and population density.
"""
AAAcounts = pd.read_csv('AAAcounts.csv')

AAAcounts.set_index(AAAcounts['county'], inplace=True)
AAAcounts['density'] = 0
AAAcounts.loc[AAAcounts.index.isin(popden.index),'density'] = popden['SUBHD0401']
AAAcounts['population'] = 0
AAAcounts.loc[AAAcounts.index.isin(popden.index),'population'] = popden['HD01']

aaa_popden = AAAcounts[['RescaledCounts','density','population']]
aaa_popden.set_index(AAAcounts.index,inplace=True)

aaa_popden['counts_log'] = np.log(aaa_popden['RescaledCounts'])
aaa_popden['density_log'] = np.log(aaa_popden['density'])
aaa_popden['pop_log'] = np.log(aaa_popden['population'])

aaa_popden.info()

np.isinf(aaa_popden).sum() #check if there is infinite values
aaa_popden = aaa_popden.replace([np.inf,-np.inf],np.nan) #replace the infinite values with nan
aaa_popden = aaa_popden[aaa_popden['counts_log'].notnull()]
aaa_popden.drop([15003, 15001, 15009], inplace=True) # Drop the three outliers that didn't rescaled well

import statsmodels.api as sm
X = sm.add_constant(aaa_popden['density_log'])
results = sm.OLS(aaa_popden['counts_log'],X).fit()
results.rsquared
results.params


sns.jointplot('density_log','counts_log',data=aaa_popden, kind='reg')
plt.text(
    3,
    11.5,
    '$\\beta=%0.3f$ $[%0.3f,%0.3f]$ $r^2=%0.3f$'
    %(results.params[1],results.conf_int(alpha=0.05, cols=None).iloc[1,0],
    results.conf_int(alpha=0.05, cols=None).iloc[1,1],results.rsquared),
    fontsize=14,
    bbox=dict(boxstyle="round", alpha=0.5)
)
plt.savefig('AAArescaledCounts_density.png')





aaa_popden.info()

landcover.head(3)


"""
Visualize AAAcounts, landcover and population density with population
"""

landcover = pd.read_excel('2010_Demo&Land_Setup.xlsx',index_col='Counties')
landcover[['water','devopen','devlow','devmed','devhigh','barren','grasslands',
'herbwetlands']] = np.log(landcover[['water','devopen','devlow','devmed','devhigh',
                            'barren','grasslands','herbwetlands']])






landcover.drop(['snowice'],axis=1,inplace=True)

landcover.loc[landcover.index.isin(aaa_popden.index),'counts_log'] = aaa_popden['counts_log']
landcover.loc[landcover.index.isin(aaa_popden.index),'density_log'] = aaa_popden['density_log']
landcover.loc[landcover.index.isin(aaa_popden.index),'pop_log'] = aaa_popden['pop_log']

np.isinf(landcover).sum() #check if there is infinite values
landcover = landcover.replace([np.inf,-np.inf],np.nan) #replace the infinite values with nan
landcover = landcover[landcover['counts_log'].notnull()]

landcover.info()
#landcover.isna().sum() # Check if the dataframe has any nan
#landcover = landcover.dropna() #Drop rows with nan
landcover.to_csv('AAAcounts_landcover.csv')


# function of showing coefficient, conf_int and r-squared of OLS while graphing
def olsfunc(x, y, **kws):
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    x = x[x.index.isin(y.index)]
    y = y[y.index.isin(x.index)]
    X = sm.add_constant(x)
    results = sm.OLS(y,X).fit()
    ax = plt.gca()
    ax.text(
        0,
        0.8,
        '$\\beta=%0.3f$\n$[%0.3f,%0.3f]$\n$r^2=%0.3f$'
        %(results.params[1],results.conf_int(alpha=0.05, cols=None).iloc[1,0],
        results.conf_int(alpha=0.05, cols=None).iloc[1,1],results.rsquared),
        transform=ax.transAxes, fontsize=20
    )






landcover = pd.read_csv('AAAcounts_landcover.csv', index_col='Counties')


# Matrix scatterplot of population_log as x-axis
#                       popden,AAAcounts and landcovers as y-axis
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[0:4], kind='reg',
                 markers=".", size=5)
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
g.map(olsfunc)
plt.show()

plt.clf()
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[4:8], kind='reg',
                 markers=".", size=5)
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
g.map(olsfunc)
plt.show()

plt.clf()
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[8:12], kind='reg',
                 markers=".", size=5)
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
g.map(olsfunc)
plt.show()

plt.clf()
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[[12,13,14,16]], kind='reg',
                 markers=".", size=5)
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
g.map(olsfunc)
plt.show()

plt.clf()
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[15], kind='reg',
                 markers=".", size=5)
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
g.map(olsfunc)
plt.show()

plt.savefig('pairplot_PopLandcover4.png')


""" Deprecated
g = sns.pairplot(landcover,x_vars=['pop_log'],y_vars=landcover.columns[12:18],kind='reg',
                 markers=".", size=5)
plt.gca().set_xlim((9,18))
plt.rcParams["axes.labelsize"] = 13
g.map(olsfunc)
plt.show()
plt.savefig('pairplot_PopLandcover4.png')

"""


"""
Visualize landcover and population density with AAAcounts
"""
plt.clf()
p = sns.pairplot(landcover, x_vars=landcover.drop(['pop_log','counts_log'], axis=1).columns,
                y_vars=['counts_log'], size=5, kind='reg', markers='.')
p.map(olsfunc)
plt.show()
plt.savefig('pairplot_CountsLandcover.png')





"""
Distance analysis to the scaling laws
y-axis is AAAcounts
x-axis is 16 variables including landcover and popden
"""
## Pearson
landcover['counts_log']= landcover.pop('counts_log') #move counts_log and pop_log to the end

d={}
for i in range(16):
    temp = landcover.iloc[:,[i,16,17]]
    temp.dropna(inplace=True)

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp['counts_log'],X).fit()
    temp['D_counts'] = results.resid

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp.iloc[:,0],X).fit()
    temp['D_variable'] = results.resid

    d["p{0}".format(i)] = sns.jointplot(x=temp.D_variable, y=temp.D_counts, kind='hex', height=8)
    d["p{0}".format(i)].set_axis_labels("Distance_%s"%landcover.columns[i],
                                        'Distance_AAAcounts',
                                        fontsize=15) # Change xlabel name within loop
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



## Linear Regression
landcover['counts_log']= landcover.pop('counts_log') #move counts_log and pop_log to the end

rawpvalues = np.zeros((16,1))
for i in range(16):
    temp = landcover.iloc[:,[i,16,17]]
    temp.dropna(inplace=True)

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp['counts_log'],X).fit()
    temp['D_counts'] = results.resid

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp.iloc[:,0],X).fit()
    temp['D_variable'] = results.resid

    X = sm.add_constant(temp['D_counts'])
    results = sm.OLS(temp['D_variable'],X).fit()
    rawpvalues[i] = results.pvalues.iloc[1]

from statsmodels.stats.multitest import multipletests
adjustedpvalues = multipletests(rawpvalues.flatten().transpose(), alpha=0.05, method='bonferroni')[1]


d={}
for j in range(16):
    temp = landcover.iloc[:,[j,16,17]]
    temp.dropna(inplace=True)

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp['counts_log'],X).fit()
    temp['D_counts'] = results.resid

    X = sm.add_constant(temp['pop_log'])
    results = sm.OLS(temp.iloc[:,0],X).fit()
    temp['D_variable'] = results.resid

    X = sm.add_constant(temp['D_counts'])
    results = sm.OLS(temp['D_variable'],X).fit()

    d["p{0}".format(j)] = sns.jointplot(x=temp.D_counts, y=temp.D_variable, kind='hex', height=8)
    sns.regplot(temp.D_counts, temp.D_variable, ax=d["p{0}".format(j)].ax_joint, scatter=False)
    ax = plt.gca()
    ax.text(
        0,
        0.8,
        '$\\beta=%0.3f$\n$pvalue=%0.5f$\n$r^2=%0.3f$'
        %(results.params[1],adjustedpvalues[j],results.rsquared),
        transform=ax.transAxes, fontsize=20
    )

    d["p{0}".format(j)].set_axis_labels('Distance_AAAcounts',"Distance_%s"%landcover.columns[j],
                                        fontsize=15) # Change xlabel name within loop
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























"""
Deprecated
"""

"""
Run with similar population
"""
# Population distribution
landcover = pd.read_csv('AAAcounts_landcover.csv')
population = population['pop_log']

# Divide data into four quantiles
landcover['popGroup'] = pd.qcut(population,4, labels=False)
landcover.set_index('Counties', inplace=True)

landcover_Q0 = landcover[landcover['popGroup'] == 0]
landcover_Q1 = landcover[landcover['popGroup'] == 1]
landcover_Q2 = landcover[landcover['popGroup'] == 2]
landcover_Q3 = landcover[landcover['popGroup'] == 3]
# Histogram distribution of population with quantile line
plt.hist(population, bins=50)
plt.axvline(x=10.383, color='k')
plt.axvline(x=10.894, color='k')
plt.axvline(x=11.794, color='k')


landcover.head(3)






p = sns.pairplot(landcover_Q3, x_vars=landcover.drop('popGroup', axis=1).columns,
                y_vars=['counts_log'], kind='reg')
p.map(olsfunc)
plt.savefig('pairplot_CountsLandcover.png')







landcover['pop_log'].min()

np.exp(9.86)
