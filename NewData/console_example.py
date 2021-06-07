import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt

def show_example():
	conn = sql.connect('weather.db') # подключение к БД
	weather = pd.read_sql('SELECT * FROM df_merged', conn) # выбрать все данные
	weather['index'] = pd.to_datetime(weather['index'], dayfirst=True)
	weather.index = weather['index']
	weather = weather.drop(columns=['index'])
	print(weather.head(5))
	print('Средняя температура по месяцам')
	print(weather.groupby(by=weather.index.month).mean().head()['T'])
	# weather.groupby(by=weather.index.month).mean().head()['T'].plot(kind='bar')
	# plt.show()
	# weather.groupby(by=weather.index.year).mean().head()['T'].plot(kind='bar')
	# plt.show()

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
	weather.groupby(by=weather.index.month).mean().head()['T'].plot(kind='bar', ax=axes[0])
	weather.groupby(by=weather.index.year).mean().head()['T'].plot(kind='bar', ax=axes[1])
	plt.show()

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def prase_args():
	import argparse

	parser = argparse.ArgumentParser(description='Optional app description')
	parser.add_argument('--opt_arg', type=int, help='An optional integer argument')
	args = parser.parse_args()

	print("Argument values:")
	print(args.opt_arg)


if __name__ == "__main__":
	suppress_qt_warnings()
	prase_args()

	show_example()

	import sys
	input_ = input()
	if input_ == 'q':
		sys.exit()