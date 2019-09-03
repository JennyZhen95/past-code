import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import seaborn as sns


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

def olsfunc_multigroups(x, y, z, data, **kws):
    d = data[[x, y, z]]
    X = sm.add_constant(d.loc[d[z] == 0, x])
    results0 = sm.OLS(d.loc[d[z] == 0, y], X).fit()

    X = sm.add_constant(d.loc[d[z] == 1, x])
    results1 = sm.OLS(d.loc[d[z] == 1, y], X).fit()

    ax = plt.gca()
    ax.text(
        0.01,
        0.8,
        '$\\beta_0=%0.3f$ $r^2_0=%0.3f$ $pvalue_0=%0.5f$\n$\\beta_1=%0.3f$ $r^2_1=%0.3f$ $pvalue_1=%0.5f$'
        % (results0.params[1], results0.rsquared, results0.pvalues.iloc[1], results1.params[1], results1.rsquared,
           results1.pvalues.iloc[1]),
        transform=ax.transAxes, fontsize=10
    )




def broadgroup_plotting_scaling_distance_pop(x, data):
    data.dropna(subset=[x], inplace=True)

    sns.lmplot(x, 'counts_log', data=data, hue='broadgroup', scatter_kws={"s": 5})
    olsfunc_multigroups(x, 'counts_log', 'broadgroup', data=data)
    plt.show()

    sns.lmplot('pop_log', 'counts_log', data=data, hue='broadgroup', scatter_kws={"s": 5})
    olsfunc_multigroups('pop_log', 'counts_log', 'broadgroup', data=data)
    plt.show()

    sns.lmplot('pop_log', x, data=data, hue='broadgroup', scatter_kws={"s": 5})
    olsfunc_multigroups('pop_log', x, 'broadgroup', data=data)
    plt.show()

    temp = data[['counts_log', x, 'pop_log', 'broadgroup']]
    temp.dropna(inplace=True)

    X = sm.add_constant(temp.loc[temp.broadgroup == 0, 'pop_log'])
    results = sm.OLS(temp.loc[temp.broadgroup == 0, 'counts_log'], X).fit()
    temp.loc[temp.broadgroup == 0, 'D_counts'] = results.resid

    X = sm.add_constant(temp.loc[temp.broadgroup == 1, 'pop_log'])
    results = sm.OLS(temp.loc[temp.broadgroup == 1, 'counts_log'], X).fit()
    temp.loc[temp.broadgroup == 1, 'D_counts'] = results.resid

    X = sm.add_constant(temp.loc[temp.broadgroup == 0, 'pop_log'])
    results = sm.OLS(temp.loc[temp.broadgroup == 0, x], X).fit()
    temp.loc[temp.broadgroup == 0, x] = results.resid

    X = sm.add_constant(temp.loc[temp.broadgroup == 1, 'pop_log'])
    results = sm.OLS(temp.loc[temp.broadgroup == 1, x], X).fit()
    temp.loc[temp.broadgroup == 1, x] = results.resid

    p = sns.lmplot('D_counts', x, data=temp, hue='broadgroup', scatter_kws={"s": 5})
    olsfunc_multigroups('D_counts', x, 'broadgroup', data=temp)
    plt.ylabel("D_%s"%x)
    plt.show()

    return(temp)



def broadgroup_distance_analysis_pop_resampling(x, data):
    ## Group1
    group1 = data[data.broadgroup == 1]
    group1.reset_index(col_fill='county', inplace=True)
    replic1 = np.array([np.random.choice(group1.shape[0], 500, replace=True) for _ in range(1000)])

    beta1 = np.zeros([1000, 1])
    beta1_se = np.zeros([1000, 1])
    for i in range(1000):
        df = group1.loc[replic1[i, :]]
        X = sm.add_constant(df['D_counts'])
        results = sm.OLS(df[x], X).fit()
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
    plt.clf()

    ## Group0
    group0 = data.loc[data.broadgroup == 0]
    group0.reset_index(col_fill='county', inplace=True)
    replic0 = np.array([np.random.choice(group0.shape[0], 500, replace=True) for _ in range(1000)])

    beta0 = np.zeros([1000, 1])
    beta0_se = np.zeros([1000, 1])
    for i in range(1000):
        df = group0.loc[replic0[i, :]]
        X = sm.add_constant(df['D_counts'])
        results = sm.OLS(df[x], X).fit()
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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(beta.index[:101], beta.beta1[:101], color='coral', label='group1')
    ax.plot(beta.index[:101], beta.beta0[:101], label='group0')
    ax.fill_between(beta.index[:101], beta.beta1[:101] - beta.beta1_se[:101],
                    beta.beta1[:101] + beta.beta1_se[:101],
                    color='coral', alpha=0.1)
    ax.fill_between(beta.index[:101], beta.beta0[:101] - beta.beta0_se[:101],
                    beta.beta0[:101] + beta.beta0_se[:101],
                    alpha=0.1)
    ax.axhline(y=beta1.mean(), color='coral', linestyle='--')
    ax.axhline(y=beta0.mean(), linestyle='--')
    ax.legend(loc='best')
    ax.set_xlabel('sample')
    ax.set_ylabel(r'$\bar \beta$')
    ax.set_title('First 100 samples')

    plt.show()
    plt.clf()


    return(beta0, beta1)

def cohen_d(x,y):
    nx = x.shape[0]
    ny = y.shape[0]
    d = (x.mean()-y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2)/(nx+ny-2))
    return(d)

def plot_multiple_distribution_with_cohen_d(x,y):
    ES = cohen_d(x,y)
    plt.clf()
    sns.distplot(x, hist_kws=dict(edgecolor="k", linewidth=0.2), color='b', label='group0')
    sns.distplot(y, hist_kws=dict(edgecolor="k", linewidth=0.2), color='coral', label='group1')
    plt.legend()
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    ax.plot([x.mean(), y.mean()], [ymax - 0.3, ymax - 0.3], color='k')
    ax.annotate('d = %0.3f' % (ES), xy=(min(x.mean(), y.mean()), ymax - 0.15), va="top", fontsize=15)
    ax = plt.gca()
    ax.text(
        0.01,
        0.8,
        r'$\bar \beta_0 = %0.3f$'
        "\n"
        r'$\bar \beta_1 = %0.3f$'
        % (x.mean(), y.mean()),
        transform=ax.transAxes, fontsize=10
    )
    plt.show()
