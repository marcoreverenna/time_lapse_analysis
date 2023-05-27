#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for obtain a new metholody for analysing time-lapse embryo development and predict abnormalities due to segregation errors

__authors__ = Matteo Figliuzzi, Marco Reverenna
__copyright__ = Copyright 2022-2023
__version__ = ---
__maintainer__ = Marco Reverenna
__email__ = marcoreverenna@gmail.com
__status__ = Dev
"""


# importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
import csv

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import PCA
from scipy import stats


# define a class Analyzer
class TimeLapseAnalyzer():

    def __init__(self, PCA_LL_TH=-50):
        """
        """
        self.list_time_features = ['tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8',
                                   't9', 'tM', 'tSB', 'tB', 'tEB']
        self.list_blasto_features = ['tB', 'tEB']# ['tB','tEB','tHB']
        self.list_maxmin_variables = ['cc3_imp_z', 'cc2_imp_z', 's2_imp_z', 's3_imp_z']

        self.imputed_times = ['tPNa_imp',
                              'tPNf_imp', 't2_imp', 't3_imp', 't4_imp', 't5_imp', 't6_imp', 't7_imp', 't8_imp',
                              't9_imp',
                              'tM_imp', 'tSB_imp', 'tB_imp', 'tEB_imp',
                              'cc2_imp', 'cc3_imp', 's2_imp', 's3_imp', 'blast_imp', 'blast1_imp']

        self.ind_times = [
            'tPNf_ind', 't2_ind', 't3_ind', 't4_ind', 't5_ind', 't6_ind', 't7_ind', 't8_ind', 't9_ind',
            'tSB_ind', 'tB_ind', 'tEB_ind']

        self.z_times = [t + '_z' for t in self.imputed_times]

        self.dict_imp_avg = None
        self.dict_imp_sigma = None

        self.PCA_LL_TH = PCA_LL_TH

    def dump_model(self, model_path):
        """
        """
        # save the model to disk
        filename_model = model_path + '.sav'
        filename_features = model_path + '.feature_list'
        filename_impavg = model_path + '.imp.avg'
        filename_avg = model_path + '.avg'
        filename_pca = model_path + '.pca'
        filename_sigma = model_path + '.sigma'
        pickle.dump(self.clf, open(filename_model, 'wb'))
        pickle.dump(self.pca, open(filename_pca, 'wb'))

        with open(filename_features, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(self.model_feature_list)

        self.avg_times.to_csv(filename_avg)
        self.dict_imp_avg.to_csv(filename_impavg)
        self.dict_imp_sigma.to_csv(filename_sigma)

    def load_model(self, model_path):
        """
        """
        # load the model from disk
        filename_model = model_path + '.sav'
        filename_pca = model_path + '.pca'
        filename_features = model_path + '.feature_list'
        filename_impavg = model_path + '.imp.avg'
        filename_avg = model_path + '.avg'
        filename_sigma = model_path + '.sigma'
        loaded_model = pickle.load(open(filename_model, 'rb'))
        self.clf = loaded_model
        self.pca = pickle.load(open(filename_pca, 'rb'))

        loaded_model_feature_list = []
        with open(filename_features, 'r') as f:
            # using csv.writer method from CSV package
            csv_reader = csv.reader(f)
            for row in csv_reader:
                loaded_model_feature_list = loaded_model_feature_list + row
        self.model_feature_list = loaded_model_feature_list

        self.dict_imp_avg = pd.read_csv(filename_impavg, squeeze=True, index_col=0)
        self.dict_imp_sigma = pd.read_csv(filename_sigma, squeeze=True, index_col=0)
        self.avg_times = pd.read_csv(filename_avg, squeeze=True, index_col=0)

    def train(self, df_train, model_feature_list, target, subset=[], exclude_qc=True):
        """
        """
        self.df_train = df_train.copy()
        self.df_train = get_binary_target(self.df_train)

        self.model_feature_list = model_feature_list
        self.target = target

        self.df_train_processed = self.process_data(self.df_train, train=True)

        if len(subset) > 0:
            df_sel = self.df_train_processed[self.df_train_processed['class'].isin(subset)]
        else:
            df_sel = self.df_train_processed
        if exclude_qc:
            df_sel = df_sel[df_sel['FLAG_QC'] == False]
        X_train = df_sel[self.model_feature_list]
        y_train = df_sel[self.target]
        self.clf = LogisticRegression(max_iter=5000)
        self.clf.fit(X_train, y_train)

    def predict(self, df):
        """
        """
        predictions = self.clf.predict_proba(df[self.model_feature_list])[:, 1]
        return predictions

    def process_data(self, df, QC_MAX_MISSING=5, train=True):
        """
        """
        # quality controll
        df['QC_NUM_MISSING'] = df[self.list_time_features].isna().transpose().sum()
        df['FLAG_QC_NUM_MISSING'] = df['QC_NUM_MISSING'] > QC_MAX_MISSING
        df['FLAG_QC_NOT_MONOTONY'] = df[self.list_time_features].apply(lambda x: check_monotony(x), axis=1) == False
        df['FLAG_QC_MISSING_BLASTO'] = (df[self.list_blasto_features].isna().sum(axis=1) == len(self.list_blasto_features))

        # imputation
        if train == True:
            self.avg_times = get_avg_times(df, self.list_time_features)

        df = impute_times(df, self.list_time_features, self.avg_times)

        # feature engineering
        df = get_intervals(df)
        df = get_ind_times(df)

        if train == True:
            self.dict_imp_avg = get_avg_times(df, self.imputed_times)
            self.dict_imp_sigma = get_sigma_times(df, self.imputed_times)
            df = get_stardardized(df, self.imputed_times, self.dict_imp_avg, self.dict_imp_sigma)
            # self.pca = pca_fit(df,self.ind_times)
            self.pca = pca_fit(df, self.z_times)
        else:
            df = get_stardardized(df, self.imputed_times, self.dict_imp_avg, self.dict_imp_sigma)

        # df = pca_analysis(df,self.ind_times,self.pca)
        df = pca_analysis(df, self.z_times, self.pca)

        df['FLAG_QC_PCA'] = df['pca_ll'] < self.PCA_LL_TH
        df['FLAG_QC'] = df['FLAG_QC_NUM_MISSING'] | df['FLAG_QC_NOT_MONOTONY'] | df['FLAG_QC_MISSING_BLASTO'] | df['FLAG_QC_PCA']
        df['QC_REASON'] = df['FLAG_QC_NUM_MISSING'].map({True: 'too many missing times;', False: ''}) + \
                          df['FLAG_QC_MISSING_BLASTO'].map({True: 'missing blasto times;', False: ''}) + \
                          df['FLAG_QC_NOT_MONOTONY'].map({True: 'not monotonous times;', False: ''}) + \
                          df['FLAG_QC_PCA'].map({True: 'low pca likelihood;', False: ''})

        df = get_maxmin_delay(df, self.list_maxmin_variables)
        return df


def func_imputation(x, i, avg_times):
    """
    return linearly imputed time
    """
    if pd.isna(x.iloc[i]) is False:
        return x.iloc[i]
    else:
        l = len(x)  
        whoisnotna = np.array(1 - x.isna())  
        cardinal = np.arange(l)

        tmp1 = whoisnotna * cardinal  
        tmp2 = whoisnotna * cardinal[::-1]  

        is_pre = cardinal < i  
        is_post = cardinal > i

        pre_idx = max(is_pre * tmp1)
        post_idx = l - 1 - max(is_post * tmp2)

        pre_coeff = (avg_times[post_idx] - avg_times[i]) / (avg_times[post_idx] - avg_times[pre_idx])
        post_coeff = (avg_times[i] - avg_times[pre_idx]) / (avg_times[post_idx] - avg_times[pre_idx])
        return pre_coeff * x.iloc[pre_idx] + post_coeff * x.iloc[post_idx]


def impute_times(df, list_time_features, avg_times=None):
    """
    """
    df['t0'] = 0  
    df['tmax'] = 1000  
    time_list = ['t0'] + list_time_features + ['tmax']
    if avg_times is None:
        avg_times = df[time_list].mean()
    for time_idx in np.arange(1, len(time_list) - 1):
        df[time_list[time_idx] + '_imp'] = df[time_list].apply(lambda x: func_imputation(x, time_idx, avg_times),axis=1)

    return df


def get_avg_times(df, list_time_features):
    """
    """
    df['t0'] = 0 
    df['tmax'] = 1000 
    time_list = ['t0'] + list_time_features + ['tmax']
    avg_times = df[time_list].mean()

    return avg_times


def get_sigma_times(df, list_time_features):
    """
    """
    df['t0'] = 0  
    df['tmax'] = 1000  
    time_list = ['t0'] + list_time_features + ['tmax']
    avg_times = df[time_list].std()

    return avg_times


def check_monotony(x):
    """
    """
    return pd.Series(x).dropna().is_monotonic_increasing


def get_intervals(df):
    """
    """
    df['cc2'] = df['t3'] - df['t2']
    df['cc3'] = df['t5'] - df['t3']
    df['s2'] = df['t4'] - df['t3']
    df['s3'] = df['t8'] - df['t5']
    df['blast'] = df['tSB'] - df['t2']
    df['blast1'] = df['tB'] - df['tSB']
    df['cc2_imp'] = df['t3_imp'] - df['t2_imp']
    df['cc3_imp'] = df['t5_imp'] - df['t3_imp']
    df['s2_imp'] = df['t4_imp'] - df['t3_imp']
    df['s3_imp'] = df['t8_imp'] - df['t5_imp']
    df['blast_imp'] = df['tSB_imp'] - df['t2_imp']
    df['blast1_imp'] = df['tB_imp'] - df['tSB_imp']
    return df


def get_ind_times(df):
    """
    """
    df['tPNf_ind'] = df['tPNf_imp'] - df['tPNa_imp']
    df['t2_ind'] = df['t2_imp'] - df['tPNf_imp']
    df['t3_ind'] = df['t3_imp'] - df['t2_imp']
    df['t4_ind'] = df['t4_imp'] - df['t3_imp']
    df['t5_ind'] = df['t5_imp'] - df['t4_imp']
    df['t6_ind'] = df['t6_imp'] - df['t5_imp']
    df['t7_ind'] = df['t7_imp'] - df['t6_imp']
    df['t8_ind'] = df['t8_imp'] - df['t7_imp']
    df['t9_ind'] = df['t9_imp'] - df['t8_imp']
    df['tSB_ind'] = df['tSB_imp'] - df['tM_imp']
    df['tB_ind'] = df['tB_imp'] - df['tSB_imp']
    df['tEB_ind'] = df['tEB_imp'] - df['tB_imp']

    return df


def get_binary_target(df):
    """
    """
    df['is_euploid'] = df['class'] == 'eup'
    df['is_segmental'] = df['class'] == 'segm'
    df['is_aneuploid'] = df['class'] == 'aneup'
    df['is_segmental_aneuploid'] = df['class'] == 'segm+aneup'
    return df


def get_stardardized(df, times, dict_avg, dict_sigma):
    """
    """
    dict_Z = {}
    times = ['tPNa_imp','tPNf_imp', 't2_imp', 't3_imp', 't4_imp', 't5_imp', 't6_imp', 't7_imp', 't8_imp', 't9_imp',
             'tM_imp', 'tSB_imp', 'tB_imp', 'tEB_imp', 'cc2_imp', 'cc3_imp', 's2_imp', 's3_imp', 'blast_imp', 'blast1_imp']

    for time in times:
        valori = (df[time] - dict_avg[time]) / dict_sigma[time]
        dict_Z[time + '_z'] = valori

    df_imp_z = pd.DataFrame.from_dict(dict_Z)
    return pd.concat([df, df_imp_z], axis=1)


def get_maxmin_delay(df, list_maxmin_variables):
    """
    """
    df['ritardo_max'] = df[list_maxmin_variables].max(axis=1)
    df['ritardo_min'] = df[list_maxmin_variables].min(axis=1)
    return df


def pca_fit(df, times):
    """
    """
    pca = PCA(n_components = 2)
    dataset_pca = df[times]
    pca.fit(dataset_pca)
    return pca


def pca_analysis(df, times, pca):
    """
    """
    pca_ll = pca.score_samples(df[times])
    df['pca_ll'] = pca_ll
    pca_out = pca.transform(df[times])
    df['pca1'] = pca_out[:, 0]
    df['pca2'] = pca_out[:, 1]
    return df


def plot_trajectory_custom(df_stat, group_variable='classif_1', class1='ane', class2='seg', plotting_times=None):
    """
    """
    if plotting_times is None:
        plotting_times = ['tPB2_imp', 'tPNa_imp', 'tPNf_imp', 't2_imp', 't3_imp', 't4_imp',
                          't5_imp', 't6_imp', 't7_imp', 't8_imp', 't9_imp', 'tSC_imp', 'tM_imp',
                          'tSB_imp', 'tB_imp', 'tEB_imp', 'tHB_imp']

    df_avg = df_stat.groupby(group_variable)[plotting_times].mean().transpose()
    df_avg.head()
    df_std = df_stat.groupby(group_variable)[plotting_times].std().transpose() / np.sqrt(df_stat.shape[0])
    df_std.head()
    df_avg['cfr'] = df_avg[class2] - df_avg[class1]

    sns.lineplot(data=df_avg.loc[plotting_times], x=class1, y='cfr', label='cfr')
    plt.fill_between(df_avg[class1], df_avg['cfr'] - df_std[class2], df_avg['cfr'] + df_std[class2], alpha=0.1, color='b')

    plt.xticks(df_avg[class1], df_avg.index, rotation=90)
    plt.grid()
    plt.plot([0, 100], [0, 2.5], c='gray', ls='--')
    plt.plot([0, 100], [0, 1.0], c='gray', ls='--')
    plt.plot([0, 100], [0, 0.5], c='gray', ls='--')
    plt.text(100, 2.5, '2.5% delay')
    plt.text(100, 1, '1% delay')
    plt.text(100, 0.5, '0.5% delay')
    plt.axhline(0, c='k')
    plt.ylabel('delay (hours)')


def plot_imputation(df_imputation,
                    index_embryo,
                    outputfile=None,
                    time_list=['t0', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'tSC', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'tmax'],
                    ymax=140):
    avg_times = df_imputation[time_list].mean()

    fig, ax = plt.subplots(figsize=(12, 3))
    lns1 = sns.scatterplot(ax=ax, data=df_imputation.iloc[index_embryo, :][time_list], color='blue', label='available data')
    ax2 = ax.twinx()

    lns2 = sns.lineplot(ax = ax2,
                        data = df_imputation.iloc[index_embryo, :][['t0'] + [(t + '_imp') for t in time_list[1:]]],
                        color = 'r', label='imputed trajectory')
    ax2.set_xticklabels(time_list)
    ax3 = ax2.twinx()
    lns3 = ax3.plot(avg_times, c='k', label='typical trajectory', alpha=0.5, ls='--')
    ax.set_ylim([0, ymax])
    ax2.set_ylim([0, ymax])
    ax3.set_ylim([0, ymax])
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax.legend(loc=4)
    
    if outputfile:
        plt.savefig(outputfile)
    else:
        plt.show()


def plot_trajectory_new(df_stat, plotting_times=None, outfile=None):
    if plotting_times is None:
        plotting_times = ['tPB2_imp', 'tPNa_imp', 'tPNf_imp', 't2_imp', 't3_imp', 't4_imp',
                          't5_imp', 't6_imp', 't7_imp', 't8_imp', 't9_imp', 'tSC_imp', 'tM_imp',
                          'tSB_imp', 'tB_imp', 'tEB_imp', 'tHB_imp']

    df_avg = df_stat.groupby('class')[plotting_times].mean().transpose()
    df_avg.head()
    df_std = df_stat.groupby('class')[plotting_times].std().transpose()
    df_std.head()
    df_count = df_stat.groupby('class')[['class']].count().rename(columns={'class': 'count'})
    
    for classif in ['eup', 'aneup', 'segm', 'segm+aneup']:
        df_std[classif] = df_std[classif] / np.sqrt(df_count.loc[classif].values[0])
    
    df_avg['ane-eup'] = df_avg['aneup'] - df_avg['eup']
    df_avg['seg-eup'] = df_avg['segm'] - df_avg['eup']
    df_avg['ane_seg-eup'] = df_avg['segm+aneup'] - df_avg['eup']

    sns.lineplot(data = df_avg.loc[plotting_times], x = 'eup', y = 'seg-eup', label = 'segmental')
    plt.fill_between(df_avg['eup'], df_avg['seg-eup'] - df_std['segm'], df_avg['seg-eup'] + df_std['segm'], alpha=0.1, color='b')

    sns.lineplot(data=df_avg.loc[plotting_times], x = 'eup', y = 'ane-eup', label = 'full aneuploidies', color='r')
    plt.fill_between(df_avg['eup'], df_avg['ane-eup'] - df_std['aneup'], df_avg['ane-eup'] + df_std['aneup'], alpha=0.1, color='r')

    plt.xticks(df_avg['eup'], df_avg.index, rotation=90)
    plt.grid()
    plt.plot([0, 100], [0, 2.5], c = 'gray', ls = '--')
    plt.plot([0, 100], [0, 1.0], c = 'gray', ls = '--')
    plt.plot([0, 100], [0, 0.5], c = 'gray', ls = '--')
    plt.text(100, 2.5, '2.5% delay')
    plt.text(100, 1, '1% delay')
    plt.text(100, 0.5, '0.5% delay')
    plt.axhline(0, c = 'k')
    plt.ylabel('delay (hours)')
    #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)


def plot_siblings(df_pairs, df_processed_all):
    plotting_times = ['tPNa_imp', 'tPNf_imp', 't2_imp', 't3_imp', 't4_imp',
                      't5_imp', 't6_imp', 't7_imp', 't8_imp', 't9_imp', 'tM_imp',
                      'tSB_imp', 'tB_imp', 'tEB_imp']
    plotting_times_delta = [x + '_delta' for x in plotting_times]

    df_siblings = df_pairs[(df_pairs.class_x == 'segm') & (df_pairs.class_y == 'eup')]
    df_avg = -df_siblings[plotting_times_delta].mean().transpose()
    df_std = df_siblings[plotting_times_delta].std().transpose() / np.sqrt(df_siblings.shape[0])
    df_avg.index = [x[0:-6] for x in df_avg.index]
    df_std.index = [x[0:-6] for x in df_std.index]
    df_avg = pd.concat([df_processed_all.groupby('class')[plotting_times].mean().transpose(), df_avg], axis=1)
    df_std = pd.concat([df_processed_all.groupby('class')[plotting_times].mean().transpose(), df_std], axis=1)
    
    sns.lineplot(data = df_avg.loc[plotting_times], x = 'eup', y = 0, label = 'segmental')
    plt.fill_between(df_avg['eup'], df_avg[0] - df_std[0], df_avg[0] + df_std[0], alpha = 0.1, color = 'b')

    df_siblings = df_pairs[(df_pairs.class_x == 'eup') & (df_pairs.class_y == 'aneup')]
    df_avg = df_siblings[plotting_times_delta].mean().transpose()
    df_std = df_siblings[plotting_times_delta].std().transpose() / np.sqrt(df_siblings.shape[0])
    df_avg.index = [x[0:-6] for x in df_avg.index]
    df_std.index = [x[0:-6] for x in df_std.index]
    df_avg = pd.concat([df_processed_all.groupby('class')[plotting_times].mean().transpose(), df_avg], axis=1)
    df_std = pd.concat([df_processed_all.groupby('class')[plotting_times].mean().transpose(), df_std], axis=1)
    sns.lineplot(data=df_avg.loc[plotting_times], x = 'eup', y = 0, label = 'aneuploid', color = 'r')
    plt.fill_between(df_avg['eup'], df_avg[0] - df_std[0], df_avg[0] + df_std[0], alpha = 0.1, color = 'r')

    plt.xticks(df_avg['eup'], df_avg.index, rotation=90)
    plt.grid()
    plt.plot([0, 100], [0, 2.5], c = 'gray', ls = '--')
    plt.plot([0, 100], [0, 1.0], c = 'gray', ls = '--')
    plt.plot([0, 100], [0, 0.5], c = 'gray', ls = '--')
    plt.text(100, 2.5, '2.5% delay')
    plt.text(100, 1, '1% delay')
    plt.text(100, 0.5, '0.5% delay')
    plt.axhline(0, c = 'k')
    plt.ylabel('delay (hours)')