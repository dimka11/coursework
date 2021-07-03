# all from Console_example.ipynb

import sqlite3 as sql
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import shap
import re
import requests
from bs4 import BeautifulSoup
import warnings

from utils import *

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# download html class
class Downloader:
    def __init__(self, start_year, end_year, station_id, db_name='weather.db'):
        self.start_year = start_year
        self.end_year = end_year
        self.station_id = station_id
        self.db_name = db_name

    def download(self, save_in_db=True):
        start_year = self.start_year
        end_year = self.end_year
        station_id = self.station_id

        m_df = pd.DataFrame()

        for year in range(start_year, end_year + 1):
            for month in range(1, 12 + 1):
                url = f"https://www.ogimet.com/cgi-bin/gsynres?ind={station_id}&lang=en&decoded=yes&ndays=31&ano={year}&mes={month}&day=1"
                soup = BeautifulSoup(requests.get(url).content, "html.parser")

                header = [th.get_text(strip=True) for th in soup.thead.select("tr")[0].select("th")]

                all_data = []
                for row in soup.thead.select("tr")[1:]:
                    tds = [td.get_text(strip=True) for td in row.select("td")[:-3]]
                    tds.insert(0, tds.pop(0) + " " + tds.pop(0))

                    for td in row.select("td")[-3:]:
                        img = td.select_one("img[onmouseover]")
                        if img:
                            tds.append(re.search(r"'([^']+)'", img["onmouseover"]).group(1))
                        else:
                            tds.append("-")

                    all_data.append(tds)

                df = pd.DataFrame(all_data, columns=header)
                #                 print(df)
                #                 df.to_csv(f"{year}_{month}.csv", index=False)
                df = df.reindex(index=df.index[::-1])
                m_df = m_df.append(df)
        m_df.to_csv(f'weather_data_{station_id}')
        if save_in_db:
            conn = sql.connect(self.db_name)  # подключение к БД
            m_df.to_sql(f'weather_data_{station_id}', conn)

        return m_df

    @staticmethod
    def save_to_db(filename, table_name, db_name):
        conn = sql.connect(db_name)
        data = pd.read_csv(filename, skiprows=6, sep=';', index_col=0)
        data.index = pd.to_datetime(data.index, dayfirst=True)
        data.to_sql(filename, table_name, conn)


# prepare data class
class PreProcessData:
    def __init__(self, data):
        self.m_df = data

    def clean_om(self):
        m_df = self.m_df
        # drop dublicates
        m_df = m_df.loc[~m_df.index.duplicated(keep='first')]

        # replace '-----' to nan
        m_df['T(C)'].replace('-----', np.nan, inplace=True)
        m_df['Td(C)'].replace('-----', np.nan, inplace=True)
        m_df['Tmax(C)'].replace('-----', np.nan, inplace=True)
        m_df['Tmin(C)'].replace('-----', np.nan, inplace=True)

        m_df['T(C)'] = m_df['T(C)'].astype(float)
        m_df['Td(C)'] = m_df['Td(C)'].astype(float)
        m_df['Tmax(C)'] = m_df['Tmax(C)'].astype(float)
        m_df['Tmin(C)'] = m_df['Tmin(C)'].astype(float)

        # Clear incorrect
        m_df['T(C)'].replace(-90.4, np.nan, inplace=True)

        # replace NaN to mean of 2 nb
        m_df['T(C)'] = m_df['T(C)'].where(m_df['T(C)'].values == 999, other=(m_df['T(C)'].fillna(method='ffill') + m_df[
            'T(C)'].fillna(method='bfill')) / 2)

        m_df = m_df.shift(periods=2)
        m_df = m_df.iloc[2:]

        return m_df


# Очистка и соединение в один датасет
class MergeData:
    def __init__(self, file_list):
        self.file_list = file_list
        self.df_omsk = None
        self.df_airport = None
        self.df_kal = None
        self.df_sher = None
        self.df_sar = None

        self.df_merged = None

    def load_merge(self, save=True):
        u = Utils()

        self.df_omsk = pd.read_csv(self.file_list[0], skiprows=6, sep=';', index_col=0)
        self.df_omsk.index = pd.to_datetime(self.df_omsk.index, dayfirst=True)
        self.df_omsk = u.inp_prep(self.df_omsk)

        self.df_airport = pd.read_csv(self.file_list[1], skiprows=6, sep=';', index_col=0)
        self.df_airport.index = pd.to_datetime(self.df_airport.index, dayfirst=True)
        self.df_airport = u.inp_prep(self.df_airport, col_drop_list=['Unnamed: 13', "W'W'", 'ff10'])

        # to one df
        df_airport_ = self.df_airport['2012-11':'2021-02']
        df_omsk_ = self.df_omsk['2012-11':'2021-02']

        df_airport_ = df_airport_[['T', 'DD', 'Ff', 'P0', 'c']]
        df_airport_ = df_airport_.rename(columns={'T': 'T_a', 'DD': 'DD_a', 'Ff': 'Ff_a', 'P0': 'P0_a', 'c': 'c_a'})

        df_omsk_ = df_omsk_[['T', 'Po', 'DD', 'Ff', 'N', 'Nh', 'Cl', 'U', 'Cm', 'Ch', 'H',
                             'RRR']]  # Nh, количество всех наблюдающихся облаков Cl или, при отсутствии облаков Cl, количество всех наблюдающихся облаков Cm

        df_omsk_airport_m = pd.merge(df_airport_, df_omsk_, how='inner', left_index=True, right_index=True)

        df_omsk_airport_m1 = df_omsk_airport_m[
            ['T', 'T_a', 'DD', 'Ff', 'N', 'Nh', 'Cl', 'Po', 'U', 'Ch', 'Cm', 'H', 'RRR']]
        df_omsk_airport_m1 = u.conv_wind(df_omsk_airport_m1)
        df_omsk_airport_m1 = u.conv_cloud(df_omsk_airport_m1)
        df_omsk_airport_m1 = u.conv_cloud(df_omsk_airport_m1, col_name='Nh')

        df_omsk_airport_winter = u.select_winter_p(df_omsk_airport_m1)

        df_omsk_airport_winter['diff'] = df_omsk_airport_winter['T'] - df_omsk_airport_winter['T_a']
        df_omsk_airport_winter['abs_diff'] = abs(df_omsk_airport_winter['T'] - df_omsk_airport_winter['T_a'])

        # load kal
        df_kal = pd.read_csv(self.file_list[2], skiprows=6, sep=';', index_col=0)
        df_kal.index = pd.to_datetime(df_kal.index, dayfirst=True)
        df_kal = df_kal[['T']]
        self.df_kal = df_kal.rename(columns={'T': 'T_kal'})

        # load sherb
        df_sher = pd.read_csv('RP5/28791.01.02.2005.04.05.2021.1.0.0.ru.utf8.00000000.csv', skiprows=6, sep=';',
                              index_col=0)
        df_sher.index = pd.to_datetime(df_sher.index, dayfirst=True)
        df_sher = df_sher[['T']]
        self.df_sher = df_sher.rename(columns={'T': 'T_sher'})

        # load sar
        df_sar = pd.read_csv('RP5/28598.01.02.2005.04.05.2021.1.0.0.ru.utf8.00000000.csv', skiprows=6, sep=';',
                             index_col=0)
        df_sar.index = pd.to_datetime(df_sar.index, dayfirst=True)
        df_sar = df_sar[['T']]
        self.df_sar = df_sar.rename(columns={'T': 'T_sar'})

        # # to one df
        df_three = pd.concat([self.df_kal, self.df_sher, self.df_sar], axis=1)

        # select winter periods
        df_three_w = u.select_winter_p(df_three)
        df_three_w['three_mean'] = df_three_w[['T_kal', 'T_sher', 'T_sar']].mean(axis=1, skipna=True)

        # add omsk and airport
        df_merged = df_omsk_airport_winter.merge(df_three_w, how='inner', left_index=True, right_index=True)

        df_merged['diff_3'] = df_merged['T'] - df_merged['three_mean']
        df_merged['abs_diff_3'] = abs(df_merged['T'] - df_merged['three_mean'])

        df_merged1 = df_merged.drop(columns=['T_kal', 'T_sher', 'T_sar'])
        self.df_merged = df_merged1
        # save a new df
        # df_merged.to_csv('df_merged')

    def save_to_db(self, db_name='weather.db'):
        conn = sql.connect(db_name)  # подключение к БД
        self.df_merged.to_sql('df_merged', conn)


class Slicer():
    def __getitem__(self, slice_):
        return slice_


class Prediction:
    cat_cols = None
    test_period = None
    df_merged_w_preds = None

    def __init__(self, df, min_t=-1, train_period=Slicer()['2012':'2019-03'], test_period=Slicer()['2019-11':],
                 drop_cols=['diff', 'abs_diff', 'diff_3', 'abs_diff_3', 'T_a', 'T_kal', 'T_sher', 'T_sar',
                            'three_mean'], target='diff', cat_cols=['DD', 'DD_', 'Cl', 'Cm', 'Ch', 'H', 'RRR']):

        self.cat_cols = cat_cols
        self.test_period = test_period

        self.df_merged_c = df.copy()  # load data
        self.df_merged_c = self.df_merged_c[self.df_merged_c['abs_diff'] > min_t]

        self.df_merged_c['Cl'] = self.df_merged_c['Cl'].fillna(0)
        self.df_merged_c['Ch'] = self.df_merged_c['Ch'].fillna(0)
        self.df_merged_c['Cm'] = self.df_merged_c['Cm'].fillna(0)
        self.df_merged_c['H'] = self.df_merged_c['H'].fillna(0)
        self.df_merged_c['RRR'] = self.df_merged_c['RRR'].fillna(0)

        self.train_data = self.df_merged_c[train_period]
        self.train_data = self.train_data.drop(columns=drop_cols)
        self.eval_data = self.df_merged_c[test_period]
        self.eval_data = self.eval_data.drop(columns=drop_cols)
        self.train_labels = self.df_merged_c[train_period][target]

        is_cat = (self.train_data.dtypes != float)
        cat_features_index = np.where(is_cat)[0]

        self.pool = Pool(self.train_data, self.train_labels, cat_features=cat_features_index,
                         feature_names=list(self.train_data.columns))

    def fit(self):
        self.model = CatBoostRegressor(iterations=1000,
                                       # {'depth': 10, 'l2_leaf_reg': 5, 'iterations': 1000, 'learning_rate': 0.3}
                                       task_type="GPU",
                                       devices='0:1',
                                       depth=10,
                                       l2_leaf_reg=5,
                                       learning_rate=0.3)

        #         display(self.train_data.info())

        self.model.fit(self.train_data,
                       self.train_labels,
                       verbose=False,
                       cat_features=self.cat_cols)

    def plot_feature_imp(self):
        plt.bar(self.train_data.columns, self.model.get_feature_importance())
        plt.show()

    def plot_tree(self, tree_idx_=0):
        self.model.plot_tree(tree_idx_, pool=self.pool)

    def plot_shap(self, type_=0, log_s=False):
        shap.initjs()
        shap_values = self.model.get_feature_importance(
            Pool(self.train_data, self.train_labels, cat_features=self.cat_cols), type='ShapValues')
        self.model.get_feature_importance()
        if type_ == 0:  # Summary
            if log_s == False:
                shap.summary_plot(shap_values, self.train_data)
            else:
                shap.summary_plot(shap_values, self.train_data, use_log_scale=True)
        if type_ == 1:  # force_plot (one)
            expected_value = shap_values[0, -1]
            shap_values = shap_values[:, :-1]
            # visualize the first prediction's explanation
            shap.force_plot(expected_value, shap_values[0, :], self.train_data.iloc[0, :])
        if type_ == 2:  # force_plot
            shap.force_plot(expected_value, shap_values[:250], self.train_data[:250])
        if type_ == 3:  # dependence plot
            shap.dependence_plot("Ff", shap_values, self.train_data)

    def mape_vectorized_v2(self, x):
        a = x['diff']
        b = x['diff_p']
        mask = a != 0
        return (np.fabs((a - b)) / a)[mask].mean()

    def mean_absolute_percentage_error(self, x):
        y_true = x['diff']
        y_pred = x['diff_p']
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def predict(self, plot_with=False):
        preds = self.model.predict(self.eval_data)

        mape_vectorized_v2 = self.mape_vectorized_v2
        mean_absolute_percentage_error = self.mean_absolute_percentage_error

        eps = 1e-6
        df_merged_w_preds = self.df_merged_c[self.test_period].copy()
        df_merged_w_preds['diff_p'] = pd.Series(preds, index=df_merged_w_preds.index)
        df_merged_w_preds['error'] = abs(df_merged_w_preds['diff'] - df_merged_w_preds['diff_p'])
        df_merged_w_preds['error rel'] = abs(df_merged_w_preds['error'] / df_merged_w_preds['diff'] + 1e-6)
        df_merged_w_preds['error rel lg'] = np.log(
            abs(df_merged_w_preds['error'] / df_merged_w_preds['diff'] + 1e-6) * 10 + 1)
        df_merged_w_preds['mape'] = df_merged_w_preds.apply(mean_absolute_percentage_error, axis=1)

        self.df_merged_w_preds = df_merged_w_preds

        print(df_merged_w_preds[['diff', 'diff_p', 'error', 'error rel', 'error rel lg']])

        if plot_with:
            PlotData.plot_data(df_merged_w_preds)

    def mean_error(self, t_diff=1):
        print(self.df_merged_w_preds['error'][self.df_merged_w_preds['abs_diff'] > t_diff].mean())

    def mean_diff(self, t_diff=1):
        print(abs(self.df_merged_w_preds['diff'][self.df_merged_w_preds['abs_diff'] > t_diff]).mean())


class PlotData:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def plot_data(data):
        # bar_plot
        df_merged_w_preds = data.copy()
        df_merged_w_preds = df_merged_w_preds[df_merged_w_preds['abs_diff'] >= 1]

        #     e1 = df_merged_w_preds['error'][df_merged_w_preds['T'] <-5].mean()
        #     ad1 = abs(df_merged_w_preds['diff'][df_merged_w_preds['T'] <-5]).mean()

        #     e2 = df_merged_w_preds['error'][df_merged_w_preds['T'] <-15].mean()
        #     ad2 = abs(df_merged_w_preds['diff'][df_merged_w_preds['T'] <-15]).mean()

        #     e3 = df_merged_w_preds['error'][df_merged_w_preds['T'] <-20].mean()
        #     ad3 = abs(df_merged_w_preds['diff'][df_merged_w_preds['T'] <-20]).mean()

        #     e4 = df_merged_w_preds['error'][df_merged_w_preds['T'] <-25].mean()
        #     ad4 = abs(df_merged_w_preds['diff'][df_merged_w_preds['T'] <-25]).mean()

        e1 = df_merged_w_preds['error'][df_merged_w_preds['DD_'] == 'North'].mean()
        ad1 = abs(df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'North']).mean()
        ad1_ = df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'North'].mean()

        e2 = df_merged_w_preds['error'][df_merged_w_preds['DD_'] == 'South'].mean()
        ad2 = abs(df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'South']).mean()
        ad2_ = df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'South'].mean()

        e3 = df_merged_w_preds['error'][df_merged_w_preds['DD_'] == 'East'].mean()
        ad3 = abs(df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'East']).mean()
        ad3_ = df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'East'].mean()

        e4 = df_merged_w_preds['error'][df_merged_w_preds['DD_'] == 'West'].mean()
        ad4 = abs(df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'West']).mean()
        ad4_ = df_merged_w_preds['diff'][df_merged_w_preds['DD_'] == 'West'].mean()

        data1 = [e1, e2, e3, e4]
        data2 = [ad1, ad2, ad3, ad4]
        data3 = [ad1_, ad2_, ad3_, ad4_]

        labels = ['', 'North', '', 'South', '', 'East', '', 'West']
        X = np.arange(4)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xticklabels(labels)
        ax.bar(X + 0.00, data1, color='b', width=0.1)
        ax.bar(X + 0.1, data2, color='g', width=0.1)
        ax.bar(X + 0.2, data3, color='r', width=0.1)
        ax.legend(['error', 'abs actual diff', 'actual diff'], loc='upper right')


# get data from db
class WeatherData:
    conn = None
    weather = None

    #     @enforce_type_hints
    def __init__(self, db_name: str) -> pd.core.frame.DataFrame:
        self.conn = sql.connect(db_name)

    def select_all(self, table_name: str):
        self.weather = pd.read_sql(f'SELECT * FROM {table_name}', self.conn)
        self.weather['index'] = pd.to_datetime(self.weather['index'], dayfirst=True)
        self.weather.index = self.weather['index']
        self.weather = self.weather.drop(columns=['index'])
        return self.weather

    # start: str, end: str
    def select_period(self, table_name, start, end):
        self.weather = pd.read_sql(f'SELECT * FROM {table_name}', self.conn)
        self.weather['index'] = pd.to_datetime(self.weather['index'], dayfirst=True)
        self.weather.index = self.weather['index']
        self.weather = self.weather.drop(columns=['index'])
        return self.weather.loc[start:end]


    def get_from_db(self, table_name):
        data = pd.read_sql(f'SELECT * FROM {table_name}', self.conn)
        data.to_csv(f'{table_name}.csv')

    def mean_by(self, month: bool = True, year: bool = False) -> pd.core.frame.DataFrame:
        if self.weather is None:
            raise Exception('You must call select_all for set table name first')
        if month == True:
            return self.weather.groupby(by=self.weather.index.month).mean().head()['T']
        if year == True:
            return self.weather.groupby(by=self.weather.index.year).mean().head()['T']

    @staticmethod
    def bar_chart(data: pd.core.frame.DataFrame) -> None:
        print(data.plot(kind='bar'))

    @staticmethod
    def print_head(table: pd.core.frame.DataFrame, count: int = 5) -> None:
        print(table.head(count))


# Агрегация
class AggData:
    def __init__(self, df_merged, w_calm=False):
        self.df_merged = df_merged
        if not w_calm:
            self.df_merged = df_merged[df_merged['DD_'] != 'Calm']

    def all_data(self, min_diff=0, only_temp_below=30, with_plot=False, with_kde=False, kde_abs_diff=False, Nh=-1,
                 Po=-1, oblast=False):
        df_merged = self.df_merged
        diff = None
        abs_diff = None
        if not oblast:
            diff = 'diff'
            abs_diff = 'abs_diff'
        else:
            diff = 'diff_3'
            abs_diff = 'abs_diff_3'

        if only_temp_below != 30:
            df_merged = df_merged[df_merged['T'] <= only_temp_below]
        if Nh != -1:
            df_merged = df_merged[df_merged['Nh'] >= Nh]
        if Po != -1:
            df_merged = df_merged[df_merged['Po'] <= Po]

        if min_diff != 0:
            data = df_merged[df_merged[abs_diff] >= min_diff].groupby('DD_', as_index=False)[[diff, abs_diff]].mean()
            data_ = df_merged[df_merged[abs_diff] >= min_diff]
        else:
            data = df_merged.groupby('DD_', as_index=False)[[diff, abs_diff]].mean()
            data_ = df_merged
        self.show(data)
        if with_plot:
            self.show_pie(data_)
        if with_kde:
            if kde_abs_diff:
                self.show_kde(data_, abs_diff=True)
            else:
                self.show_kde(data_, abs_diff=False)

        print('rows: ', len(df_merged.index))

    @staticmethod
    def show(data):
        print(data.head())

    @staticmethod
    def show_pie(data):
        data['DD_'].value_counts().plot(kind='pie')
        plt.show()

    def show_kde(self, data, abs_diff_=False, oblast=False):
        if not oblast:
            diff = 'diff'
            abs_diff = 'abs_diff'
        else:
            diff = 'diff_3'
            abs_diff = 'abs_diff_3'

        if abs_diff_:
            fig, ax = plt.subplots()
            data[(data['DD_'] == 'East')][abs_diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'North')][abs_diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'South')][abs_diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'West')][abs_diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            ax.legend(["East", "North", "South", "West"])
            plt.show()
        else:
            fig, ax = plt.subplots()
            data[(data['DD_'] == 'East')][diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'North')][diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'South')][diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            data[(data['DD_'] == 'West')][diff].plot.kde(bw_method=0.1, xlim=(-5, 5))
            ax.legend(["East", "North", "South", "West"])
            plt.show()

    def plot_wind_dist(self):
        v1 = self.df_merged['DD_'].value_counts()
        v1.name = ''
        v1 = v1.rename(
            {'Calm': 'Штиль', 'East': 'Восточный', 'North': 'Северный', 'South': 'Южный', 'West': 'Западный'})
        v1.plot(kind='pie')

    # period = string: year-month
    def get_corr(self, period=None):
        df_merged_ = self.df_merged.copy()
        df_merged_['DD_'] = df_merged_['DD_'].astype('category').cat.codes
        if period is None:
            print(df_merged_.corr())
        else:
            print(df_merged_[period].corr())

    # period = string: year-month
    def get_heatmap(self, period='2014-12'):
        df_merged_ = self.df_merged.copy()
        df_merged_['DD_'] = df_merged_['DD_'].astype('category').cat.codes

        sns.heatmap(df_merged_[period].corr(), annot=True)
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.show()
