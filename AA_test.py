import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import itertools

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

hw = pd.read_csv('hw_aa.csv', sep='\t')

def metric_fpr(df_x, df_y, metric_col, n_sim, n_s_perc, n_s_min, estimator, *args, **kwargs):
    stat_res = {
        'aa': {
            'pvalue': [],
            'mu_x': [],
            'mu_y': []
        },
        'fpr': {
            'fpr_95': 0
        }
    }

    for sim in range(n_sim):

        # по умолчанию берем % n_s_perc наблюдений от исходной, но не более n_s_min
        x = df_x[metric_col].sample(int(min(n_s_min, len(df_x) * n_s_perc)), replace=False).values
        y = df_y[metric_col].sample(int(min(n_s_min, len(df_y) * n_s_perc)), replace=False).values

        if estimator == 'prop':
            counts = np.array([sum(x), sum(y)])
            nobs = np.array([len(x), len(y)])
            stat, pvalue = proportions_ztest(counts, nobs, *args, **kwargs)

        if estimator == 'ttest':
            stat, pvalue = stats.ttest_ind(x, y, *args, **kwargs)

        stat_res['aa']['pvalue'].append(pvalue)
        stat_res['aa']['mu_x'].append(np.mean(x))
        stat_res['aa']['mu_y'].append(np.mean(y))

    stat_res['fpr']['fpr_95'] = float(sum(np.array(stat_res['aa']['pvalue']) <= 0.05) / n_sim)

    return stat_res

def fpr_report(df, metric_col, variant_col, group_col, n_sim, n_s_perc, n_s_min, estimator, *args, **kwargs):
    list_fpr = []
    list_group = list(pd.unique(df[group_col]))

    for v in range(len(list_group)):
        df_x = df[(df[variant_col] == 0) & (df[group_col] == list_group[v])]
        df_y = df[(df[variant_col] == 1) & (df[group_col] == list_group[v])]
        cr_x = sum(df_x[metric]) / len(df_x)
        cr_y = sum(df_y[metric]) / len(df_y)

        fpr = {}
        fpr = metric_fpr(
            df_x=df_x,
            df_y=df_y,
            metric_col=metric,
            n_sim=n_sim,
            n_s_perc=n_s_perc,
            n_s_min=n_s_min,
            estimator=estimator, *args, **kwargs
        )
        is_fpr = (fpr['fpr']['fpr_95'] <= 0.05)
        list_fpr.append([list_group[v], cr_x, cr_y, fpr['fpr']['fpr_95'], is_fpr])

    report = pd.DataFrame.from_records(list_fpr, columns=['group', 'cr_x', 'cr_y', 'fpr_95', 'is_fpr'])

    return report

n_sim = 1000
n_s_perc = 0.9
n_s_min = 1000
metric = 'purchase'
variant = 'experimentVariant'
group = 'version'

res = fpr_report(
    df = hw,
    metric_col = metric,
    variant_col = variant,
    group_col = group,
    n_sim = n_sim,
    n_s_perc = n_s_perc,
    n_s_min = n_s_min,
    estimator = 'ttest'
)
res