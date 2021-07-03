import os
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# Script utils for clean and preprocess / label data
class Utils:
    def __init__(self):
        pass

    @staticmethod
    def inp_prep(df_o, col_drop_list=['P','Pa','ff3' , 'ff10', 'tR', 'E', 'Tg', 'E', 'sss', "E'", 'W1', 'W2', 'Unnamed: 29'], info=True, reverse=True):
        #clean cols
        dfa = df_o.drop(columns=col_drop_list)

        # replace NaN to mean of 2 nb
        dfa['T'] = dfa['T'].where(dfa['T'].values == 999, other=(dfa['T'].fillna(method='ffill') + dfa['T'].fillna(method='bfill'))/2)
        # datetimeindex
        dfa.index = pd.to_datetime(dfa.index, dayfirst=True)
        if info is True:
             dfa.info()
        if reverse is True:
            # reverse
            dfa = dfa.iloc[::-1]

        return dfa

    # Wind to 4 cats
    @staticmethod
    def conv_wind(dfa, add_column=True):
        m_df4_ = dfa['DD'].replace(['Ветер, дующий с запада', 'Ветер, дующий с юго-запада', 'Ветер, дующий с западо-юго-запада', 'Ветер, дующий с западо-северо-запада'],'West')
        m_df4_ = m_df4_.replace(['Ветер, дующий с севера', 'Ветер, дующий с северо-запада', 'Ветер, дующий с северо-северо-востока', 'Ветер, дующий с северо-северо-запада'],'North')
        m_df4_ = m_df4_.replace(['Ветер, дующий с востока', 'Ветер, дующий с северо-востока', 'Ветер, дующий с востоко-юго-востока', 'Ветер, дующий с востоко-северо-востока'],'East')
        m_df4_ = m_df4_.replace(['Ветер, дующий с юга', 'Ветер, дующий с юго-востока', 'Ветер, дующий с юго-юго-запада', 'Ветер, дующий с юго-юго-востока'],'South')
        m_df4_ = m_df4_.replace(['Штиль, безветрие'],'Calm')
        m_df4 = dfa.copy()
        if add_column:
            m_df4['DD_'] = m_df4_
            m_df4['DD_'].unique()
        else:
            m_df4['DD'] = m_df4_
            m_df4['DD'].unique()
        return m_df4

    # clouds values to float type
    @staticmethod
    def conv_cloud(dfa, col_name='N'):
        m_df4_ = dfa[col_name].replace(['70 – 80%.'],75)
        m_df4_ = m_df4_.replace(['90  или более, но не 100%'],90)
        m_df4_ = m_df4_.replace(['60%.'],60)
        m_df4_ = m_df4_.replace(['100%.'], 100)
        m_df4_ = m_df4_.replace(['Облаков нет.'],0)
        m_df4_ = m_df4_.replace(['40%.'],40)
        m_df4_ = m_df4_.replace(['20–30%.'],25)
        m_df4_ = m_df4_.replace(['10%  или менее, но не 0'],10)
        m_df4_ = m_df4_.replace(['50%.'],50)
        m_df4_ = m_df4_.replace(['Небо не видно из-за тумана и/или других метеорологических явлений.'],101)
        m_df4 = dfa.copy()
        m_df4[col_name] = m_df4_
        m_df4[col_name].unique()
        return m_df4

    # Conv cl to cat
    @staticmethod
    def cl_cat(cl_series):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        return le.fit_transform(cl_series)


    #precip values to float type
    @staticmethod
    def conv_precip(dfa):
        m_df4_ = dfa['RRR'].replace(['Осадков нет'],0)
        m_df4_ = m_df4_.replace(['Следы осадков'],0.1)
        m_df4_ = m_df4_.astype(float)
        m_df4 = dfa.copy()
        m_df4['RRR'] = m_df4_
        return m_df4

    #conv_wind_label
    @staticmethod
    def conv_wind_label(dfa, col_name='DD'):
        m_df4_ = dfa[col_name].replace(['West'],1)
        m_df4_ = m_df4_.replace(['North'],2)
        m_df4_ = m_df4_.replace(['East'],3)
        m_df4_ = m_df4_.replace(['South'],4)
        m_df4_ = m_df4_.replace(['Calm'],0)
        m_df4 = dfa.copy()
        m_df4[col_name] = m_df4_
        m_df4[col_name].unique()
        return m_df4

    # select winter periods:
    @staticmethod
    def select_winter_p(df):
        range_1 = df.loc['2005-11':'2006-02']
        range_2 = df.loc['2006-11':'2007-02']
        range_3 = df.loc['2007-11':'2008-02']
        range_4 = df.loc['2008-11':'2009-02']
        range_5 = df.loc['2009-11':'2010-02']
        range_6 = df.loc['2010-11':'2011-02']
        range_7 = df.loc['2011-11':'2012-02']
        range_8 = df.loc['2012-11':'2013-02']
        range_9 = df.loc['2013-11':'2014-02']
        range_10 = df.loc['2014-11':'2015-02']
        range_11 = df.loc['2015-11':'2016-02']
        range_12 = df.loc['2016-11':'2017-02']
        range_13 = df.loc['2017-11':'2018-02']
        range_14 = df.loc['2018-11':'2019-02']
        range_15 = df.loc['2019-11':'2020-02']
        range_16 = df.loc['2020-11':'2021-02']

        return pd.concat([range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8, range_9, range_10, range_11, range_12, range_13, range_14, range_15, range_16])