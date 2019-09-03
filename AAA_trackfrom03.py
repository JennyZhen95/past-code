import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

"""
Autism: dx1 starts with 2990
ADD, ADHD: 3140
"""
baseline = pd.read_csv('mentaldata/kid03.csv')  # 13million claims and 643619 individuals

# Take out DX1 that is AAA
AAA_baseline = baseline[(baseline['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (baseline['DX1'].apply(lambda x: str(x)[:4]) == '3140')]['ENROLID']  # 541 individuals with AAA in DX1
baseline.drop(AAA_baseline.index, inplace=True)

# Take out DX2 that is AAA
AAA_baseline = baseline[(baseline['DX2'].apply(lambda x: str(x)[:4]) == '2990') |
                        (baseline['DX2'].apply(lambda x: str(x)[:4]) == '3140')]['ENROLID']  # 87 individuals with AAA in DX2
baseline.drop(AAA_baseline.index, inplace=True)

# 643594 individuals with age 0-2
baseline = pd.read_csv('baseline2003_all claims.csv')
len(np.unique(baseline['ENROLID']))
baseline_ID = np.unique(baseline['ENROLID'])
baseline.to_csv('baseline2003_all claims.csv')

"""
Study how many kids developed AAA
2004: 1347 developed AAA
2005: 1526 developed AAA
2006: 1835 developed AAA
2007: 2580 developed AAA
2008: 3407 developed AAA
2009: 3704 developed AAA
2010: 2770 developed AAA

"""
aaa04 = pd.read_csv('mentaldata/mental_cleaned04.csv')
aaa04 = aaa04[(aaa04['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa04['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa04 = aaa04[aaa04['ENROLID'].isin(baseline_ID)]
aaa04.to_csv('aaa04.csv')
aaa04 = pd.read_csv('aaa04.csv')
baseline_ID_rest = list(set(baseline_ID) - set(np.unique(aaa04['ENROLID'])))


aaa05 = pd.read_csv('mentaldata/mental_cleaned05.csv')
aaa05 = aaa05[(aaa05['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa05['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa05 = aaa05[aaa05['ENROLID'].isin(baseline_ID_rest)]
aaa05.to_csv('aaa05.csv')
aaa05 = pd.read_csv('aaa05.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa05['ENROLID'])))

aaa06 = pd.read_csv('mentaldata/mental_cleaned06.csv')
aaa06 = aaa06[(aaa06['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa06['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa06 = aaa06[aaa06['ENROLID'].isin(baseline_ID_rest)]
aaa06.to_csv('aaa06.csv')
aaa06 = pd.read_csv('aaa06.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa06['ENROLID'])))

aaa07 = pd.read_csv('mentaldata/mental_cleaned07.csv')
aaa07 = aaa07[(aaa07['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa07['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa07 = aaa07[aaa07['ENROLID'].isin(baseline_ID_rest)]
aaa07.to_csv('aaa07.csv')
aaa07 = pd.read_csv('aaa07.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa07['ENROLID'])))

aaa08 = pd.read_csv('mentaldata/mental_cleaned08.csv')
aaa08 = aaa08[(aaa08['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa08['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa08 = aaa08[aaa08['ENROLID'].isin(baseline_ID_rest)]
aaa08.to_csv('aaa08.csv')
aaa08 = pd.read_csv('aaa08.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa08['ENROLID'])))

aaa09 = pd.read_csv('mental_cleaned09.csv')
aaa09 = aaa09[(aaa09['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa09['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa09 = aaa09[aaa09['ENROLID'].isin(baseline_ID_rest)]
aaa09.to_csv('aaa09.csv')
aaa09 = pd.read_csv('aaa09.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa09['ENROLID'])))

aaa10 = pd.read_csv('mental10_cleaned_new.csv')
aaa10 = aaa10[(aaa10['DX1'].apply(lambda x: str(x)[:4]) == '2990') |
                        (aaa10['DX1'].apply(lambda x: str(x)[:4]) == '3140')]
aaa10 = aaa10[aaa10['ENROLID'].isin(baseline_ID_rest)]
aaa10.to_csv('aaa10.csv')
aaa10 = pd.read_csv('aaa10.csv')
baseline_ID_rest = list(set(baseline_ID_rest) - set(np.unique(aaa09['ENROLID'])))


"""
Combine 2004-2010 newly developed AAA. Keep ENROLID, AGE, DX1, MSA, SEX, EMPCTY, YEAR.
"""
aaa04['YEAR'] = '2004'
aaa05['YEAR'] = '2005'
aaa06['YEAR'] = '2006'
aaa07['YEAR'] = '2007'
aaa08['YEAR'] = '2008'
aaa09['YEAR'] = '2009'
aaa10['YEAR'] = '2010'

new_aaa = pd.DataFrame()
new_aaa = new_aaa.append(aaa04[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa05[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa06[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa07[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa08[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa09[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])
new_aaa = new_aaa.append(aaa10[['ENROLID', 'AGE', 'DX1', 'MSA', 'SEX', 'EMPCTY', 'YEAR']])

new_aaa = pd.DataFrame(new_aaa)


"""
Separate AD(H)D and Autism. Value counts for each county. 
"""
# Autism 2990
new_autism = new_aaa[new_aaa['DX1'].apply(lambda x: str(x)[:4]) == '2990']
new_autism.drop_duplicates(subset='ENROLID', inplace=True) # Drop duplicated ENROLID since all claims here are generally related to autism.
new_autism.to_csv('new_autism.csv')

# AD(H)D 3140
new_add = new_aaa[new_aaa['DX1'].apply(lambda x: str(x)[:4]) == '3140']
new_add.drop_duplicates(subset='ENROLID', inplace=True)
new_add.to_csv('new_add.csv')


"""
Value counts for each county
"""
new_add = pd.read_csv('new_add.csv')
new_autism = pd.read_csv('new_autism.csv')
# AD(H)D
add_cty = pd.DataFrame()
add_cty = add_cty.append(new_add['EMPCTY'].value_counts()).transpose()
add_cty.index = add_cty.index.astype(np.int64)


"""
Individual claims with AAA track. 643594 claims.
"""
baseline.drop_duplicates(subset='ENROLID', inplace=True)
baseline.drop(['DX1', 'DX2'], axis=1, inplace=True)
baseline['ADD'] = 0
baseline['ADD'][baseline['ENROLID'].isin(new_add['ENROLID'])] = 1
baseline.sort_values(by='ENROLID', inplace=True)
new_add.sort_values(by='ENROLID', inplace=True)
baseline['ADD_year'][baseline['ENROLID'].isin(new_add['ENROLID'])] = np.int64(new_add['YEAR'])

new_autism.sort_values(by='ENROLID', inplace=True)
baseline['Autism'] = 0
baseline['Autism'][baseline['ENROLID'].isin(new_autism['ENROLID'])] = 1
baseline['Autism_year'] = 0
baseline['Autism_year'][baseline['ENROLID'].isin(new_autism['ENROLID'])] = np.int64(new_autism['YEAR'])
len(baseline)


"""
Rescaling
"""
insured = pd.read_csv('kids_insured.csv', encoding='latin1')
insured.drop(0, inplace=True)
insured.set_index('GEO.id2', inplace = True)
insured.index = insured.index.astype(int)
insured = insured[['HD01_VD04','HD01_VD07','HD01_VD32','HD01_VD35']]
insured = insured.astype(int)
insured['count'] = insured['HD01_VD04'] + insured['HD01_VD07'] + insured['HD01_VD32'] + insured['HD01_VD35']

MScount = pd.read_csv('Kid_Age_10.csv')
MScount.drop(0, inplace=True)
MScount.set_index('EMPCTY', inplace=True)
MScount.index = MScount.index.astype(int)

add_cty.rename(columns={'EMPCTY': "raw"}, inplace=True)
add_cty['insuredbyMS'] = 0
add_cty.loc[add_cty.index.isin(MScount.index),'insuredbyMS'] = MScount['_FREQ_']

add_cty['totalinsured'] = 0
add_cty.loc[add_cty.index.isin(insured.index),'totalinsured'] = insured['count']

add_cty['rescaled'] = add_cty['totalinsured']*add_cty['raw']/add_cty['insuredbyMS']
add_cty = add_cty[add_cty['totalinsured'] != 0]

landcover = pd.read_csv('AAAcounts_landcover.csv')
landcover.drop(columns='counts_log', inplace=True)
add_cty = pd.concat([add_cty, landcover.set_index('Counties')], axis=1, join='inner')

add_cty['rescaled_log'] = np.log(add_cty['rescaled'])
add_cty['raw_log'] = np.log(add_cty['raw']) #1246 entries



"""
Check rescaling
"""
plt.clf()
sns.lmplot('raw_log','rescaled_log', data=add_cty[add_cty['raw_log'] > 3], scatter_kws={"s": 3})
ax = plt.gca()
ax.set_title('AD(H)D Rescaling')
olsfunc(add_cty['raw_log'], add_cty['rescaled_log'])
plt.show()


"""
popden_log vs counts_log
"""
plt.clf()
plt.scatter(add_cty['rescaled_log'], add_cty['density_log'], s=3)
plt.xlabel('rescaled_log')
plt.ylabel('density_log')
plt.show()