import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
from os import environ
import argparse
import sys
from data_work import *


class ConsoleInterface:
    def __init__(self):
        pass

    def suppress_qt_warnings(self):
        environ["QT_DEVICE_PIXEL_RATIO"] = "0"
        environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        environ["QT_SCREEN_SCALE_FACTORS"] = "1"
        environ["QT_SCALE_FACTOR"] = "1"

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Optional app description')
        parser.add_argument('--opt_arg', type=int, help='An optional integer argument')
        args = parser.parse_args()

        print("Argument values:")
        print(args.opt_arg)

    @staticmethod
    def input():
        print('Please input command: q mb h')
        input_ = input()
        if input_ == 'q':
            sys.exit()
        if input_ == 'mb':
            return 'mb'
        if input_ == 'h':
            print('print q for exit')

    def print_text(self, data):
        print(data)

    @staticmethod
    def print_text_first(n, data):
        print(data.head(n))

    @staticmethod
    def show_diagram(data):
        pass


class GetData:
    def __init__(self, db_name):
        self.conn = sql.connect(db_name)  # подключение к БД

    def get_data(self, table_name):
        weather = pd.read_sql(f'SELECT * FROM {table_name}', self.conn)  # выбрать все данные
        weather['index'] = pd.to_datetime(weather['index'], dayfirst=True)
        weather.index = weather['index']
        weather = weather.drop(columns=['index'])
        return weather

    def get_from_db(self, db_name, table_name):
        data = pd.read_sql(f'SELECT * FROM {table_name}', self.conn)
        data.to_csv(f'{table_name}.csv')


class AggData:
    def __init__(self, data):
        self.weather = data

    def mean_by_month(self, print_text=False, show_diagram=False):
        weather = self.weather
        if print_text:
            print(weather.groupby(by=weather.index.month).mean().head()['T'])
        if show_diagram:
            weather.groupby(by=weather.index.month).mean().head()['T'].plot(kind='bar')
            plt.show()
            weather.groupby(by=weather.index.year).mean().head()['T'].plot(kind='bar')
            plt.show()
        else:
            return weather.groupby(by=weather.index.month).mean().head()['T']


class MyDataWork:
    @staticmethod
    def show_example(ci):
        get_data = GetData('weather.db')
        weather = get_data.get_data('df_merged')
        ci.print_text_first(5, weather)  # Первые 5 строк

        print('Средняя температура по месяцам')
        agg_data = AggData(weather)
        agg_data.mean_by_month(print_text=True, show_diagram=True)


# Запуск
if __name__ == "__main__":
    # ci.parse_args()

    ci = ConsoleInterface()
    ci.suppress_qt_warnings()

    result = ci.input()
    if result == 'mb':
        MyDataWork.show_example(ci)

    ci.input()
