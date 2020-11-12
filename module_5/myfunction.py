# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import re 
import math
import plotly
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import time

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from tabulate import tabulate
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from scipy.stats import ttest_ind
from itertools import combinations
from collections import Counter
from IPython.display import HTML, display

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

pd.set_option('display.max_columns', None)

# %% [code]

def drop_dublle(data, columns):
    data.drop_duplicates(subset=columns, keep='first', inplace=True)
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После удаления дублирующих строк по идентификатору '{}'".
                format(columns))
    return data

def season_from_year(date):
    if date == 0:
        return 'empty'
    else:
        month = date.month
        if month == 12 or month < 3:
            return ['winter']
        elif month < 6:
            return ['spring']
        elif month < 9:
            return ['summer']
        else:
            return ['autumn']
    
def print_report(rows_total, rows_train, columns, text):
    print(text +', Датасет содержит признаков - {}; строк - {}, из них {} - train.'.
         format(columns, rows_total, rows_train))

# соберем из списка списков одноуровневый список
def list_extend(list_of_lists):
    result=[]
    for lst in list_of_lists:
        result.extend(lst)
    return result

# вывод таблицы со сводкой о датасете
def brief_summary(data, columnwidth):
    brief_columns = ['Признак', '# заполненных', 'тип данных', '% заполнения', '# пропусков',
                     '# уникальных', 'диапазон значений / примеры'] 
    df = pd.DataFrame(columns = brief_columns)
    columns_list = data.columns.to_list()
    for x in range(len(columns_list)):
        column = columns_list[x]
        count = len(data[data[column].isnull()])
        res = str(data[column].iloc[0])+'<br>'+str(data[column].iloc[10])+'<br>'+str(data[column].iloc[20])
        df.loc[x] = [
            '<b>'+column+'</b>',
            len(data) - count,
            str(data[column].dtype),
            round((1 - len(data[data[column].isnull()])/len(data))*100, 1),
            count,
            data[column].nunique(),
            res
        ]
    
    fig = go.Figure(data=[go.Table(
            columnwidth = columnwidth,
            header=dict(values=brief_columns,
                        fill_color='paleturquoise',
                        align='center',
                       font=dict(size=12)),
            cells=dict(values=[df[i] for i in brief_columns],
                       fill_color='lavender',
                       align=['left'] + ['center']*5 + ['left'],
                      height=60))
        ])
    fig.update_layout(margin = dict(l=50, r=50, t=50, b=20)
                     ,width = 800)

    fig.show()

# Добавление столбца с количеством ресторанов в сети
def calc_count_in_chain(data):    
    df = pd.DataFrame(data['Restaurant_id'].value_counts()).reset_index()
    df.columns = ['Restaurant_id', 'count_in_chain']
    if 'count_in_chain' in data.columns.to_list():
        data.drop(['count_in_chain'], axis=1, inplace = True)
    data = pd.merge(data, df, on = 'Restaurant_id')
    data['name_chain'] = data['count_in_chain'].apply(lambda x: ['single'] if x<2 else ['several'] if x<5 else ['many'])
    text = ('Добавлены признаки count_in_chain с количеством ресторанов в объединении,'
            'к которому имеет отношение текущий ресторан и признак сети/объединения - code_chain')
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), text)
    return data
    
# вывод структуры уникальных и сетевых ресторанов 
def view_count_in_chain(data):
    df = data['Restaurant_id'].value_counts()
    layout = go.Layout(
          autosize=False,
          width=1200,
          height=350)
    fig = go.Figure(layout = layout)
    fig.add_trace(go.Histogram(x = df[df == 1], opacity=.75, name = 'уникальные рестораны'))
    fig.add_trace(go.Histogram(x = df[(df > 1) & (df <= 4)], opacity=.75, name = 'ресторанные объединения'))
    fig.add_trace(go.Histogram(x = df[df > 4], opacity=.75, name = 'ресторанные сети'))
    fig.update_layout(title = 'Структура рынка уникальных и сетевых ресторанов',
                     title_x = 0.5,
                     xaxis_title = 'Ресторанов в объединении',
                     yaxis_title = '# ресторанов / объединений / сетей',
                     legend = dict(x = .8, y = 0.84,xanchor = 'center', orientation = 'v'),
                     #barmode = 'overlay',
                     margin = dict(l=100, r=50, t=50, b=20))
    fig.show()

# преобразование строк в списки
def string_to_list_distribution(data, column, new_column=True, empty_value=[]):
    # замена пропусков значением empty
    empty_values = [None, np.nan, 'nan']
    empty_values.append(empty_value)
    try: data[column].fillna('empty')
    except: a = 1
    data[column] = data[column].map(lambda x: 'empty' if x in empty_values else x)

    data['temp'] = data[column].apply(lambda x: "'"+str(x)+"'") # сервисный столбец
    
    # кодирование исходной переменной после добавления empty
    code_column = 'code_'+str(column).replace(' ', '_').lower()
    le = LabelEncoder()
    le.fit(data['temp'])
    data[code_column] = le.transform(data['temp'])
    
    # преобразование в список
    if new_column == True:
        new_column = 'list_'+str(column).replace(' ', '_').lower()
    else: new_column = column
    data[new_column] = data['temp'].str.findall(r"'(\b.*?\b)'")
    data.drop(['temp'], inplace=True, axis=1)  
    
    print("Строковый признак '{}' преобразован в список и сохранен в столбец '{}'".format(column, new_column))
    return data    

# преобразования признака Cuisine Style
def cuisine_distribution(data, column):
    # зафиксировать пустые значения в отдельной переменной
    data['empty_cuisine_style'] = data[column].apply(lambda x: 1 if 'empty' in x else 0)
    
    # посчитаем количество заявленных кухонь - признак count_cuisine_style
    data['count_cuisine_style'] = data[column].apply(lambda x: len(x)).astype('float64')

    # пропуски в столбце count_cuisine_style средним значением
    median = np.median(data[data['empty_cuisine_style'] != 1]['count_cuisine_style'])
    data['count_cuisine_style'] = data[column].apply(lambda x: median if 'empty' in x else len(x))
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг Cuisine Style")
    
    return data

# определение наиболее частых и наиболее редких вариантов признака
def rife_rare_distribution(data, column, high_percent=0.3, low_percent=0.02):
    temp_list = data[column].tolist()
    sign_set = Counter(list_extend(temp_list)).most_common()
    sign_df = pd.DataFrame(sign_set, columns=[column, '#'])
    
    high_count = sign_df['#'].sum() * high_percent
    low_count = sign_df['#'].sum() * low_percent
    
    high_list = []
    sum_count = 0
    for i in range(len(sign_df['#'])):
        if sum_count < high_count:
            if sign_df[column][i] != 'empty':
                sum_count += sign_df['#'][i]
                high_list.append(sign_df[column][i])
        else: break
    
    low_list = []
    sum_count = 0
    for i in range(len(sign_df['#']))[::-1]:
        if sum_count < low_count:
            if sign_df[column][i] != 'empty':
                sum_count += sign_df['#'][i]
                low_list.append(sign_df[column][i])
        else: break
        
    data[column.replace('list_', 'rife_')] = data.apply(lambda x: 1 if len(set(x[column])&set(high_list)) > 0 and
                                                        x[column.replace('list_', 'empty_')] != 1 else 0, axis=1)    
    data[column.replace('list_', 'rare_')] = data[column].apply(lambda x: 1 if len(set(x)&set(low_list)) > 0 else 0)
    text = ("Признак частого повторения присвоен {} вариантам в {} строках. "
            "\nПризнак редкого использования присвоен {} вариантам в {} строках").format(len(high_list), 
                                                                                       len(data[data[column.replace('list_', 'rife_')] == 1]),
                                                                                       len(low_list),
                                                                                       len(data[data[column.replace('list_', 'rare_')] == 1]))
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), text+"\nПосле добавления признаков наиболее частых и редких {}".
                format(column.replace('list_', '')))    
    return data


def localisation_cuisine_country(data):
    dict_cuisine_by_country = {
        'United Kingdom':['British','Scottish'],
        'Spain': ['Spanish', 'Mediterranean', 'Latin'],
        'France': ['French','Central European', 'Mediterranean'], 
        'Italy': ['Italian','Central European', 'Mediterranean', 'Latin'],
        'Germany': ['Dutch','German','Central European'],
        'Portugal': ['Portuguese', 'Latin'],
        'Czechia': ['Czech','Eastern European'],
        'Poland':['Polish','Eastern European'],
        'Austria': ['Austrian','Central European'],
        'Netherlands':['Central European'],
        'Belgium': ['Belgian','Eastern European'],
        'Switzerland':['Swiss','Central European'],
        'Sweden':['Scandinavian', 'Balti'],
        'Hungary':['Hungarian','Eastern European'],
        'Ireland':['Irish'],
        'Denmark':['Danish', 'Balti'],
        'Greece':['Greece', 'Mediterranean'],
        'Norway':['Scandinavian','Balti'],
        'Finland':['Scandinavian','Balti'],
        'Slovakia':['Eastern European'],
        'Luxembourg':['Eastern European'],
        'Slovenia':['Slovenian','Eastern European']
    }
    data['local_cuisine'] = data.apply(lambda x: 1 if len(set(x['list_cuisine_style'])&set(dict_cuisine_by_country[x['country']])) > 0 else 0, axis=1)
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признака соответствия кухни ресторана региону его локализации")
    
    return data

# Работа с категориальной переменной
def prep_dummies(data, column, percent=1, prefix='', limit_list = []):
    temp_list = data[column].tolist()
    sign_set = Counter(list_extend(temp_list)).most_common()
    sign_df = pd.DataFrame(sign_set, columns=[column, '#'])
    
    high_count = sign_df['#'].sum() * percent
    
    target_list = []
    sum_count = 0
    for i in range(len(sign_df['#'])):
        if sum_count < high_count:
            if sign_df[column][i] != 'empty':
                sum_count += sign_df['#'][i]
                target_list.append(sign_df[column][i])
        else: break
            
    target_drop = target_list
       
    if percent < 1:
        text = ("При подготовке признака, из перечня переменных использован {}-й процентиль. "
                 "Из первичного списка в {} значений оставлены {}".
                format(int(sum_count/sign_df['#'].sum()*100), len(sign_set), len(target_list)))
    else: text = ''
        
    if len(limit_list) > 0:
        target_list = set(target_list)&set(limit_list)
        text = ("При подготовке признака, из {} значений оставлены {}".
                format(len(target_drop), len(target_list)))
        
    for col in target_list:
        if col != 'empty':
#            if percent < 1:
#                data[column.replace('list_', 'other_')] = data[column].apply(lambda x: 1
#                                                                             if len(set(col)&(set(sign_df[column].to_list()) - set(target_list))) > 0 
#                                                                             else 0)
            data[prefix+col] = data[column].apply(lambda x: 1 if col in x else 0)

    print_report(len(data), len(data[data['sample']==1]), len(data.columns), text+"\nПосле преобразования {} в dummy-переменную".
                format(column.replace('list_', '')))    
    return data

# для Столбца: вывод на экран горизонтальной диаграммы и таблицы с количеством значений
def view_horiz_bar_n_table(data, column, new_sign='no'):
    # добавление признака с количеством объектов column
    list_counter = Counter(data[column])
    if new_sign !='no':
        data[new_sign] = data[column].apply(lambda x: list_counter[x])
    
    list_counter = list_counter.most_common()
    reversed_list_counter = list_counter[::-1]

    fig = make_subplots(rows=1, cols=2, column_widths=[6,2], specs=[[{"type": "bar"}, {"type": "table"}]])

    trace0 = go.Bar(y = [x[0] for x in reversed_list_counter]
                    ,x = [x[1] for x in reversed_list_counter]
                    ,orientation='h')

    trace1 = go.Table(header=dict(values=[column, '#'],
                    fill_color='paleturquoise',
                    align='center'),
                    cells=dict(values=[[x[0] for x in list_counter], [x[1] for x in list_counter]],
                    fill_color='lavender',
                    align='center'))

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    
    height = 170 if 35*len(list_counter) < 170 else 400 if 35*len(list_counter) > 400 else 35*len(list_counter)
            
    fig.update_layout(height= height, width=800,
                      margin = dict(l=100, r=50, t=50, b=10),
                      title = 'Распределение признака {}'.format(column),
                      title_x = 0.5)

    fig.show()
    if new_sign !='no':
        return data

# расширение признаков вокруг City с использованием внешних источников данных    
def city_expansion_features(data):
    df_city = pd.read_csv('/kaggle/input/world-cities-datasets/'+'/worldcities.csv')
    
    data['City'] = data['City'].map(lambda x: 'Porto' if x == 'Oporto' else x)
    list_counter = Counter(data['City'])
    city_info = pd.DataFrame(columns=['City', 'city_is_the_capital', 'population_city', 'country'])
    for index, city in enumerate(list_counter.keys()):
        df = df_city[df_city['city_ascii'] == city].iloc[0]
        city_info.loc[index] = [
            city,
            1 if df['capital'] == 'primary' else 0,
            int(df['population']),
            df['country']
        ]
    
    city_info = string_to_list_distribution(city_info, 'country')
    
    data = pd.merge(data, city_info, on = 'City')
    
    df_city_tourists = pd.read_excel('/kaggle/input/tourists-in-europe-city/'+'/city of Europe.xlsx')
    df_city_tourists = df_city_tourists[['City', 'Tourists']]
    df_city_tourists.columns = ['City', 'count_city_tourists']
    
    data = pd.merge(data, df_city_tourists, on = 'City')
    
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг City")
    return data

# вывод группы из четырех диаграмм и таблицы (для числовых признаков)
def view_histogram_n_boxplot(data, column):
    ds = data[column]
    if min(ds) < 0:
        add_value = math.floor(min(ds)) 
        ds_ = ds.values - add_value
    else: ds_ = ds


    dsl = np.log1p(ds_)
    
    perc25_lin = np.quantile(ds, 0.25)
    perc75_lin = np.quantile(ds, 0.75)
    IQR_lin = perc75_lin - perc25_lin

    perc25_log = np.quantile(dsl, 0.25)
    perc75_log = np.quantile(dsl, 0.75)
    IQR_log = perc75_log - perc25_log

#    data['log_'+column] = np.log1p(data[column])
#    data['log_'+column] = data['log_'+column].map(lambda x: np.nan if x == float('-inf') else x)

    fig = make_subplots(rows=1
                        ,cols=5
                        ,column_widths=[2,2,2,2,4]
                        ,specs=[[{"type": "histogram"}
                               ,{"type": "box"}
                               ,{"type": "histogram"}
                               ,{"type": "box"}
                               ,{"type": "table"}]])

    trace0 = go.Histogram(x = ds, opacity=.65)
    count_linear = go.Box(y = ds, marker_color = 'black', opacity=.5)
    trace2 = go.Histogram(x = dsl, opacity=.65, nbinsx = 8)
    count_log = go.Box(y = dsl, marker_color = 'black', opacity=.5)
    
    sign_name = ['строк', 'тип', 'значений', 'пропусков', 'первый квантиль'
                 ,'медиана', 'третий квантиль', 'межквантильный диапазон'
                 ,'нижняя граница выбросов', 'верхняя граница выбросов'
                 ,'кол-во значений за нижней границей','кол-во значений за верхней границей']
    values = [len(data)
              ,str(ds.dtype)
              ,len(ds.isna())
              ,len(data) - len(ds.isna())
              ,round(perc25_lin,1)
              ,round(np.median(ds),1)
              ,round(perc75_lin,1)
              ,round(IQR_lin,1)
              ,round(perc25_lin - 1.5*IQR_lin, 3)
              ,round(perc75_lin + 1.5*IQR_lin, 3)
              ,len(ds[(ds < perc25_lin - 1.5*IQR_lin)])
              ,len(ds[(ds > perc75_lin + 1.5*IQR_lin)])
    ]
    values_log = ['','','',''
              ,round(perc25_log,2)
              ,round(np.median(dsl),2)
              ,round(perc75_log,2)
              ,round(IQR_log,2)
              ,round(perc25_log - 1.5*IQR_log, 3)
              ,round(perc75_log + 1.5*IQR_log, 3)
              ,len(dsl[(dsl < perc25_log - 1.5*IQR_log)])
              ,len(dsl[(dsl > perc75_log + 1.5*IQR_log)])
    ]
    trace4 = go.Table(header=dict(values=['', 'linear', 'log']
                                 ,fill_color='paleturquoise'
                                 ,align='center'
                                 ,font_size=9)
                     ,cells=dict(values=[sign_name, values, values_log]
                                 ,fill_color='lavender'
                                 ,align='center'
                                 ,font_size=8))

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(count_linear, 1, 2)
    fig.append_trace(trace2, 1, 3)
    fig.append_trace(count_log, 1, 4)
    fig.append_trace(trace4, 1, 5)

    fig.update_layout(title = 'Линейные значения и логарифм признака {}, границы и количество выбросов'.format(column)
                      ,title_x = 0.5
                      ,width = 800
                      ,height = 265
                      ,margin = dict(l=20, r=20, t=50, b=0)
                      ,showlegend=False)

    fig.show()
    return data


# Сравнение параметров распределения признака column
# в зависимости от значения признака sign

def view_sign_influence_on_h_box_plot(data, column, sign):
    sign_list = np.sort(data[sign].dropna().unique())

    layout = go.Layout(autosize=False
                       ,width = 800
                       ,height = len(sign_list) * 80)
    fig = go.Figure(layout=layout)
    for i in sign_list:
        fig.add_trace(go.Box(x = data[data[sign] == i][column], name=str(i)))

    fig.update_layout(title = 'Распределения признака ' + column +
                              ' в зависимости от значения ' + sign
                      ,title_x = 0.5
                      ,margin = dict(l=100, r=100, t=30, b=30))
    iplot(fig)
    return None


# Преобразование признака Ranking
def ranking_distribution(data):
    city_list = set(data['City'].to_list())
    scaler = MinMaxScaler()

    rank = pd.DataFrame()
    for c in city_list:
        df = data[data['City'] == c][['ID_TA', 'Ranking', 'sample']]
        df['total_ranking'] = MinMaxScaler().fit_transform(np.array(df['Ranking']).reshape(-1,1))
        df['standard_ranking'] = StandardScaler().fit_transform(np.array(df['Ranking']).reshape(-1,1))
        rank = pd.concat([rank, df], sort=False)
    rank.drop(['Ranking'], axis=1, inplace=True)
    data = pd.merge(data, rank, on = ['ID_TA', 'sample'], how = 'left')
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг Ranking")
    return data

def add_ranking_distribution(data):
    mean_ranking_on_city = data.groupby(['City'])['Ranking'].mean()
    data['mean_ranking_on_city'] = data['City'].apply(lambda x: mean_ranking_on_city[x])
    data['norm_ranking_on_population'] = (data['Ranking'] - data['mean_ranking_on_city']) / (data['population_city'] / 1000)
    data['norm_ranking_on_population'].astype('float64')
    data['norm_ranking_on_tourists'] = (data['Ranking'] - data['mean_ranking_on_city']) / (data['count_city_tourists'] / data['population_city'])
    data['norm_ranking_on_tourists'].astype('float64')
    max_ranking_on_city = data.groupby(['City'])['Ranking'].max()
    data['max_ranking_on_city'] = data['City'].apply(lambda x: max_ranking_on_city[x])
    data['norm_ranking_on_max_rank'] = (data['Ranking'] - data['mean_ranking_on_city']) / data['max_ranking_on_city']
    count_city_restaurant = data.groupby(['City'])['Ranking'].count()
    data['count_city_restaurant'] = data['City'].apply(lambda x: count_city_restaurant[x])
    data['norm_ranking_on_restaurant'] = (data['Ranking'] - data['mean_ranking_on_city']) / data['count_city_restaurant']
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления дополнительных признаков вокруг Ranking")
    return data

# вывод распределения признака column по depth крупным позициям признака attribute
def view_attribute_based_distribution(data, column, attribute, depth):
    df = data[data['sample'] == 1]
    layout = go.Layout(
              autosize=False,
              width=1200,
              height=350)
    fig = go.Figure(layout = layout)
    for x in data[attribute].astype('str').value_counts()[0:depth].index:
        fig.add_trace(go.Histogram(x = df[df[attribute].astype('str') == x][column], name = x, opacity = .75, nbinsx = 200))

    fig.update_layout(title = 'Распределение признака ' + column + ' по ' + str(depth) + ' крупным значениям признака ' + attribute,
                         title_x = 0.5,
                         xaxis_title = 'Значение ' + column,
                         yaxis_title = 'Количество',
                         legend = dict(x = 1.05, y = 0.9,xanchor = 'center', orientation = 'v'),
                         barmode = 'overlay',
                         bargap=0.1,
                         margin = dict(l=100, r=50, t=50, b=20))
    fig.show()

# Преобразование Price Range
def price_distribution(data, nan_value):
    data['empty_price_range'] = pd.isna(data['Price Range']).astype('float64')
    ds = data[data['sample'] == 1]['Price Range']
    print('{} значений признака пропущены в исходном датафрейме, это составляет {} % общей выборки'.
        format(len(ds[ds.isna()]),
        round(len(ds[ds.isna()]) / len(ds) * 100, 1)))

    # Перекодировка признака
    price_dict = {'$':1, '$$ - $$$':2, '$$$$':3}
    data['price_range'] = data['Price Range'].map(lambda x: price_dict[x] if x in price_dict else nan_value)
    price_dict = {'$':'low', '$$ - $$$':'medium', '$$$$':'high'}
    data['Price Range'] = data['Price Range'].map(lambda x: price_dict[x] if x in price_dict else np.nan)
    ds = data[data['sample'] == 1]
    ds0 = ds[ds['empty_price_range'] == 0]['price_range']
    ds1 = ds[ds['empty_price_range'] == 1]['price_range']
        
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг Price Range")
    
    return data

def view_price_info(data):
    ds = data[data['sample'] == 1]
    ds0 = ds[ds['empty_price_range'] == 0]['price_range']
    ds1 = ds[ds['empty_price_range'] == 1]['price_range']
    sign_name = ['строк', 'тип', 'значений', 'пропусков', 'min', 'max', 'mean',' median']
    values = [
        len(ds),
        'object' if str(ds['Price Range'].dtype) == 'O' else str(ds['Price Range'].dtype),
        len(ds) - len(ds[ds['Price Range'].isna()]),
        len(ds[ds['Price Range'].isna()]),'','','',''
    ]
    trace0 = go.Table(header=dict(values=['', 'исходные данные'],
                    fill_color='paleturquoise',
                    align='center'),
                    cells=dict(values=[sign_name, values],
                    fill_color='lavender',
                    align='center')) 

    trace1_0 = go.Histogram(x = ds0, opacity=.65, name="исходные данные", bingroup=0, marker_color='#0000CD')
    trace1_1 = go.Histogram(x = ds1, opacity=.8, name="дополненные значения", bingroup=0, marker_color='#FF0000')
    
    ds = ds['price_range']
    values = [
        len(ds),
        'object' if str(ds.dtype) == 'O' else str(ds.dtype),
        len(ds) - len(ds[ds.isna()]),
        len(ds[ds.isna()]),
        np.min(ds),
        np.max(ds),
        round(np.mean(ds), 3),
        np.median(ds)
    ]
    trace2 = go.Table(header=dict(values=['', 'дополненные данные'],
                    fill_color='paleturquoise',
                    align='center'),
                    cells=dict(values=[sign_name, values],
                    fill_color='lavender',
                    align='center')) 

    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "table"}, {"type": "histogram"}, {"type": "table"}, {"type": "table"}]])
        
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1_0, 1, 2)
    fig.append_trace(trace1_1, 1, 2)
    fig.append_trace(trace2, 1, 3)

    fig.update_layout(barmode="stack",
                      bargap=0.1,
                      legend = dict(xanchor = 'center', orientation = 'v'),
                      height=225, margin = dict(l=20, r=20, t=0, b=0), showlegend=False)
    fig.show()

# рассчет среднего чека по городам, вывод диаграммы
def mean_price_in_city(data):
    dict_price_in_city = data.groupby('City')['price_range'].mean().to_dict()
    data['price_in_city'] = data['City'].map(dict_price_in_city)
    data['price_in_city'] = MinMaxScaler().fit_transform(np.array(data['price_in_city']).reshape(-1,1))

    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления среднего уровня цен в городах")
    return data

def view_mean_price(data):
    dict_price_in_city = data.groupby('City')['price_in_city'].mean().to_dict()
    revers_dict = dict(reversed(item) for item in dict_price_in_city.items())
    X = sorted(revers_dict.keys(), reverse = True)
    Y = [revers_dict[x] for x in X]
    fig = go.Figure()

    trace0 = go.Bar(y = Y, x = X, orientation='h')
    fig.add_trace(trace0)
    fig.update_layout(title = 'Средний уровень цен в городах (относительные значения)',
                      title_x = 0.5,
                      margin = dict(l=200, r=200, t=50, b=0))
    fig.show()


# вывод большой гистограммы распределения логарифма признака и таблицы с расчетом выбросов
def view_histogram_n_outliers(data, column, how='all', bins=0):
    ds = data[data['sample'] == 1][column]
    ds = ds[~ds.isna()]
    dsl = data[data['sample'] == 1]['log_'+column]
    dsl = dsl[~dsl.isna()]

    perc25_lin = ds.quantile(0.25)
    perc75_lin = ds.quantile(0.75)
    IQR_lin = perc75_lin - perc25_lin

    perc25_log = dsl.quantile(0.25)
    perc75_log = dsl.quantile(0.75)
    IQR_log = perc75_log - perc25_log
    
    trace_lin_0 = go.Histogram(x = ds, name = 'все значения (lin)', nbinsx=bins, opacity=.75)
    trace_lin_1 = go.Histogram(x = ds[(ds > perc75_lin + 1.5*IQR_lin) | (ds < (perc25_lin - 1.5*IQR_lin))], 
                               name = 'выбросы (lin)', nbinsx=int(bins/4), opacity=.5)
    trace_log_0 = go.Histogram(x = dsl[dsl <= perc75_log + 1.5*IQR_log], name = 'все значения (log)', nbinsx=int(bins/2), opacity=.75)
    trace_log_1 = go.Histogram(x = dsl[(dsl > perc75_log + 1.5*IQR_log) | (dsl < (perc25_log - 1.5*IQR_log))], 
                               name = 'выбросы (log)', nbinsx=int(bins/8), opacity=.5)
    
    if how == 'all':
        fig = make_subplots(rows=2, cols=2, column_widths=[1100,500], row_heights=[200,200], 
                        specs=[[{"type": "histogram"}, {"type": "table", 'rowspan': 2}],
                              [{"type": "histogram"}, None]])
        fig.add_trace(trace_lin_0,1,1)
        fig.add_trace(trace_lin_1,1,1)

        fig.add_trace(trace_log_0,2,1)
        fig.add_trace(trace_log_1,2,1)

    else:
        fig = make_subplots(rows=1, cols=2, column_widths=[1100,500], row_heights=[300], 
                        specs=[[{"type": "histogram"}, {"type": "table"}]])
        if how == 'lin':
            fig.add_trace(trace_lin_0,1,1)
            fig.add_trace(trace_lin_1,1,1)            
        elif how == 'log':
            fig.add_trace(trace_log_0,1,1)
            fig.add_trace(trace_log_1,1,1)

    sign_name = ['первый квантиль', 'медиана', 'третий квантиль', 'межквантильный диапазон', 'нижняя граница выбросов', 
                 'верхняя граница выбросов', 'кол-во значений за нижней границей','кол-во значений за верхней границей']
    values = [
            perc25_lin,
            np.median(ds),
            perc75_lin,
            IQR_lin,
            round(perc25_lin - 1.5*IQR_lin, 3),
            round(perc75_lin + 1.5*IQR_lin, 3),
            len(ds[(ds < perc25_lin - 1.5*IQR_lin)]),
            len(ds[(ds > perc75_lin + 1.5*IQR_lin)])
        ]
    values_log = [
            round(perc25_log, 3),
            round(np.median(dsl), 3),
            round(perc75_log, 3),
            round(IQR_log, 3),
            round(perc25_log - 1.5*IQR_log, 3),
            round(perc75_log + 1.5*IQR_log, 3),
            len(dsl[(dsl < perc25_log - 1.5*IQR_log)]),
            len(dsl[(dsl > perc75_log + 1.5*IQR_log)])
        ]
    fig.add_trace(go.Table(header=dict(values=['<b>Параметр</b>', '<b>lin</b>', '<b>log</b>'],
                        fill_color='paleturquoise',
                        align='center',
                        height=40),
                        cells=dict(values=[sign_name, values, values_log],
                        fill_color='lavender',
                        align=['left']+['center']*2,
                        height=50),
                        columnwidth = [250,100,100],),1,2)


    fig.update_layout(title = 'Исследование распределения и выбросов признака ' + column,
                     title_x = 0.5,
                     legend = dict(x=.5, y=.9, xanchor = 'center', orientation = 'v'),
                     barmode = 'overlay',
                     bargap=0.1,
                     margin = dict(l=50, r=50, t=50, b=0))

    fig.show()

# обработка текстовой части признака Reviews. Подсчет количества слов с позитивной и негативной окраской
def review_text_distribution(data):
    pattern_max = re.compile('((?!and)(?!for)(?!the)[A-Z|a-z]{3,})')
    data['review_words'] = data['Reviews'].apply(lambda x: pattern_max.findall(str(x).lower()))
    data['count_review_words'] = data['review_words'].apply(lambda x: len(x))
    
    data['Reviews'] = data['Reviews'].fillna('[[], []]')
    data['empty_review'] = data['Reviews'].apply(lambda x: 1 if x == '[[], []]' else 0)
    data['review_words'] = data.apply(lambda x: 'empty' if x['empty_review'] == 1 else x['review_words'], axis=1)
    
    data = string_to_list_distribution(data, 'review_words')

    positive_words, negative_words = read_positive_words()

    data['count_pos_words'] = data['list_review_words'].apply(lambda x: len(set(x)&set(positive_words)))
    data['count_neg_words'] = data['list_review_words'].apply(lambda x: len(set(x)&set(negative_words)))
    data['count_neg_words'] = data.apply(lambda x: x['count_neg_words']-1 if 'not' in x['list_review_words'] else x['count_neg_words'], axis = 1)
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг текста признака Reviews")
    return data

def read_positive_words():
    # читаем списки позитивных и негативных слов
    DATA_DIR = '/kaggle/input/opinion-lexicon-english/'
    positive_words = pd.read_csv(DATA_DIR +'positive-words.txt', skiprows=34, names=['word'])
    positive_words = positive_words['word'].to_list()
    negative_words = pd.read_csv(DATA_DIR +'neg_words.txt', 'r', encoding="ISO-8859-1", names=['word'])
    negative_words = list(set(negative_words['word'].to_list()))

    return positive_words, negative_words

# работа с датами в признаке Reviews
def data_review_distribution(data):
    pattern_max = re.compile('\[\'(\d{2}\/\d{2}/\\d{4})')
    pattern_min = re.compile('(\d{2}\/\d{2}/\\d{4})\'\]')
    data['Review_date_max'] = data['Reviews'].apply(lambda x: str(pattern_max.findall(str(x))))
    data['Review_date_min'] = data['Reviews'].apply(lambda x: str(pattern_min.findall(str(x))))
    data['review_date_count'] = data.apply(lambda x: 0 if x['Reviews'] == '[[], []]' else 
                                           1 if x['Review_date_max'] == x['Review_date_min'] else 2, axis=1)
    # Преобразуем даты в дни
    data['Review_date_max'] = (pd.datetime.now() - pd.to_datetime(data['Review_date_max'], format="['%m/%d/%Y']", errors='coerce')).dt.days
    data['Review_date_min'] = (pd.datetime.now() - pd.to_datetime(data['Review_date_min'], format="['%m/%d/%Y']", errors='coerce')).dt.days
    data['review_date_min'] = data['Review_date_min'].apply(lambda x: x - data['Review_date_min'].min() + 1)
    data['review_date_delta'] = data['Review_date_min'] - data['Review_date_max']
    data['review_date_delta'] = data['review_date_delta'].apply(lambda x: abs(x))
    # добавляем признак сезона
    #data['review_date_season'] = data['Review_date_min'].apply(lambda x: season_from_year(x))
    
    del data['Review_date_max']
    del data['Review_date_min']
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), "После добавления признаков вокруг календарных переменных Reviews")    
    return data

def show_heatmap(df):
    corrs = df.corr()
    fig = ff.create_annotated_heatmap(z=corrs.values
                                      ,x=list(corrs.columns)
                                      ,y=list(corrs.index)
                                      ,opacity=.8
                                      ,annotation_text=corrs.round(2).values
                                      ,colorscale=["rgb(0,196,95)", "rgb(224,243,220)", "rgb(69,117,220)"]
                                      ,font_colors = ["rgb(0,76,128)", "rgb(224,243,220)"]
                                      ,zmin=-1
                                      ,zmax=1
                                      ,showscale=True)
    fig.update_layout(title = 'Тепловая карта матрицы корреляций'
                      ,title_x = 0.5
                      ,width = 800
                      ,legend = dict(x = .5, xanchor = 'center', orientation = 'h')
                      ,margin = dict(l=100, r=100, t=100, b=30))

    fig.show()

# стандартизация
def normalisation(df, scaler, not_norm = [], columns_list='all'):
    if columns_list == 'all':
        columns_list = df.columns.tolist()
    for column in columns_list:
        if df[column].dtype in ['float64', 'int64'] and (column not in not_norm):
            df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1,1))
    return df

# удаление нечисловых признаков
def delete_string_sign(df):
    for column in df.columns.tolist():
        if df[column].dtype == 'object':
            del df[column]
    return df

# применение метода главных компонент
def pca_distribution(data, list_for_pca, pca_name, num_cols, list_for_save=[]):
    if len(list_for_save) == 0: list_for_save = [False for i in range(len(list_for_pca))] 
    data_temp = data.loc[data['sample'] == 1]

    for column in list_for_pca:
        data_temp[column] = MinMaxScaler().fit_transform(np.array(data_temp[column]).reshape(-1,1))
    matrix = np.array(data_temp[list_for_pca].corr())
    eig_num, eig_v = np.linalg.eig(matrix)
    res = np.zeros(len(data))

    for i in range(len(list_for_pca)):
        res = res + np.array(data[list_for_pca[i]])*eig_v.T[0][i]    
    data[pca_name] = res
    num_cols.append(pca_name)
    
    list_del = []
    for i in range(len(list_for_pca)):
        if list_for_save[i] == False or list_for_save[i] == 0:
            list_del.append(list_for_pca[i])
            num_cols.remove(i)

    data.drop(list_del, axis=1, inplace=True)
    text = ("\nПризнаки {} по методу главных компонент преобразованы в признак '{}'. \nВектор главных компонент: {}"
            "\nПосле преобразования".
            format(str(list_for_pca)[1:-1], pca_name, list(eig_v.T[0])))
    print_report(len(data), len(data[data['sample']==1]), len(data.columns), text)
    return data, num_cols

def read_dataframes():
    DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
    df_train = pd.read_csv(DATA_DIR + '/main_task.csv')
    df_test = pd.read_csv(DATA_DIR + 'kaggle_task.csv')
    sample_submission = pd.read_csv(DATA_DIR + '/sample_submission.csv')

    # ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет
    df_train['sample'] = 1 # помечаем где у нас трейн
    df_test['sample'] = 0 # помечаем где у нас тест
    df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

    data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
    return data


# Логарифмирование переменной с исправлением отрицательных значений
def get_log_sign(data, to_log_cols):
    for column in to_log_cols:
        ds = data[column]
        if min(ds) < 0:
            add_value = math.floor(min(ds)) 
            data[column] = data[column] - add_value
        data[column] = np.log1p(data[column])
    return data

        
 # Диаграмма важности признаков
def view_important_sign(data, num_cols, bin_cols, cat_cols, key_sign):
    df = data.copy()
    df.dropna(inplace=True)
    imp_num = pd.Series(f_classif(df[num_cols]
                                  ,df[key_sign])[0]
                        ,index = num_cols)
    imp_num.sort_values(inplace = True)


    imp_bin_cat = pd.Series(mutual_info_classif(df[bin_cols + cat_cols]
                                            ,df[key_sign]
                                            ,discrete_features =True)
                        ,index = bin_cols + cat_cols)
    imp_bin_cat.sort_values(inplace = True)

    imp_bin = imp_bin_cat.copy()
    imp_cat = imp_bin_cat.copy()
    for i in bin_cols + cat_cols:
        if i not in bin_cols:
            imp_bin[i] = 0
        if i not in cat_cols:
            imp_cat[i] = 0        

    fig = make_subplots(rows=1, cols=2, column_widths=[1,1], specs=[[{"type": "bar"}, {"type": "bar"}]])

    trace0 = go.Bar(name='номинативные'
                    ,y = imp_num.keys()
                    ,x = imp_num.values
                    ,orientation='h')

    trace1 = go.Bar(name='бинарные'
                    ,y = imp_bin.keys()
                    ,x = imp_bin.values
                    ,orientation='h')

    trace2 = go.Bar(name='категориальные'
                    ,y = imp_cat.keys()
                    ,x = imp_cat.values
                    ,orientation='h')

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace1, 1, 2)

    fig.update_layout(width=800
                      ,height=350
                      ,barmode='stack'
                      ,margin = dict(l=100, r=50, t=50, b=10)
                      ,title = 'Степень влияния признаков на целевую переменную'
                      ,title_x = 0.5)
    fig.show()
    

    
def get_label_encoder(data, column):
    label_encoder = LabelEncoder()
    mapped_education = pd.Series(label_encoder.fit_transform(data[column]))
    print('Переменные признака {} преобразованы в соответствии со словарем {}'.format(column, dict(enumerate(label_encoder.classes_))))
    data[column] = label_encoder.fit_transform(data[column])
    return data
    

def print_regression_metrics(y_true, y_pred, name):
#    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
#    precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'{name}: mae = {mae:.4f}, mse = {mse:.4f}, rmse = {rmse:.4f}')

    
def train_education(data, n_estimators, RANDOM_SEED):
    
    start = datetime.now()
    
    data_temp = data.drop(columns=['sample', 'client_id', 'app_date', 'default'])
    
    data_empty_ed = data_temp[data_temp['education'].isna()].drop(columns='education')
    data_full_ed = data_temp[~data_temp['education'].isna()]

    # произведена мин-макс стандартизация (за исключением списка столбцов)
    data_empty_ed = normalisation(data_empty_ed, MinMaxScaler(), ['education'])
    data_full_ed = normalisation(data_full_ed, MinMaxScaler(), ['education'])
    X = data_full_ed.drop(columns=['education'])
    y = data_full_ed['education'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = RANDOM_SEED)
    
    model = RandomForestRegressor(n_estimators=n_estimators
                                  ,verbose=1
                                  ,n_jobs=-1
                                  ,random_state = RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = np.round(model.predict(X_test))    
    
    print_regression_metrics(y_test, y_pred, 'Метрики')
    
    period = datetime.now() - start
    print('Время выполнения: {}'.format(period))
    
    return model, data_empty_ed


# Вывод набора метрик логистической регрессии
def view_metrics_LogisticRegression(model, X_train, y_train, X_test, y_test, key_sign):

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    apc = average_precision_score(y_test, y_pred)

    fig = make_subplots(rows=2
                        ,cols=2
                        ,row_heights=[4,5]
                        ,column_widths=[1,1]
                        ,vertical_spacing =.15
                        ,horizontal_spacing =.15
                        ,subplot_titles=(f"Logistic Regression (AUC={roc_auc:.5f})"
                                         ,f"Recall-precision curve (APC={apc:.5f})"
                                         ,'Check metrics')
                        ,specs=[[{"type": "scatter"}, {"type": "heatmap"}]
                               ,[{'type': 'table', 'colspan': 2}, None]]
                       )

    
    # Формирование диаграммы ROC-AUC
    trace0 = go.Scatter(x=fpr
                        ,y=tpr
                        ,mode='lines'
                        ,showlegend=False
                        ,fill='tozeroy')

    trace1 = go.Scatter(x=[0,1],y=[0,1], mode='lines', showlegend=False)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)


    '''
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    cm = np.rot90(cm, 3)
    
    trace2 = ff.create_annotated_heatmap(z=cm
                                         ,x=key_sign
                                         ,y=['False', 'True']
                                         ,opacity=.8
                                         ,annotation_text=df.values
                                         ,colorscale=["rgb(224,243,220)", "rgb(69,117,220)"]
                                         ,showscale=False)

    fig.append_trace(trace2.data[0], 1, 2)
    fig.update_xaxes(title_text="Real values", row=1, col=2)
    fig.update_yaxes(title_text="Predicted values", row=1, col=2)'''

    
    # Дтаграмма Precision-recall
    y_score = model.fit(X_train, y_train).decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    
    trace3 = go.Scatter(x=list(recall)
                        ,y=list(precision)
                        ,mode='lines'
                        ,showlegend=False
                        ,fill='tozeroy')

    trace4 = go.Scatter(x=[closest_zero_r]
                        ,y=[closest_zero_p]
                        ,mode='markers'
                        ,marker=dict(size=10
                                     ,line_width=1
                                     ,color='#ff0034')
                        ,showlegend=False)
    fig.add_trace(trace3, 1, 2)
    fig.add_trace(trace4, 1, 2)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    
    # Таблица параметров
    dict_metric = {}
    
    P = np.sum(y_test==1)
    N = np.sum(y_test==0)
    TP = np.sum((y_test==1)&(y_pred==1))
    TN = np.sum((y_test==0)&(y_pred==0))
    FP = np.sum((y_test==0)&(y_pred==1))
    FN = np.sum((y_test==1)&(y_pred==0))

    dict_metric['Positive, P'] = [P, 'Истинный {}'.format(key_sign[0])]
    dict_metric['Negative, N'] = [N, 'Истинный {}'.format(key_sign[1])]
    dict_metric['True Positive, TP'] = [TP, 'Корректная идентификация {}'.format(key_sign[0])]
    dict_metric['True Negative, TN'] = [TN, 'Корректная идентификация {}'.format(key_sign[1])]
    dict_metric['False Positive, FP'] = [FP, 'Ошибочная идентификация {}'.format(key_sign[0])]
    dict_metric['False Negative, FN'] = [FN, 'Ошибочная идентификация {}'.format(key_sign[1])]
    dict_metric['Accuracy'] = [accuracy_score(y_test, y_pred),'Точность: Accuracy=(TP+TN)/(P+N)']
    dict_metric['Precision'] = [precision_score(y_test, y_pred),'Precision = TP/(TP+FP)'] 
    dict_metric['Recall'] = [recall_score(y_test, y_pred),'Recall = TP/P']
    dict_metric['F1-score'] = [f1_score(y_test, y_pred),'Метрика, объединяющая Precision и Recall']
    dict_metric['ROC_AUC'] = [roc_auc_score(y_test, y_pred_prob),'ROC AUC Score']
    

    temp_df = pd.DataFrame.from_dict(dict_metric, orient='index', columns=['Значение', 'Описание'])

    brief_columns = ['', 'Значение', 'Описание']
    trace5 = go.Table(columnwidth = [3,2,6]
                      ,header=dict(values=brief_columns
                                   ,fill_color='paleturquoise'
                                   ,align='center'
                                   ,font=dict(size=12))
                      ,cells=dict(values=[temp_df.index
                                          ,list(temp_df.iloc[:6,0].values.round(0))
                                          +list(temp_df.iloc[6:,0].values.round(5))
                                          ,temp_df.iloc[:,1].values]
                                  ,fill_color='lavender'
                                  ,align=['left'] + ['center'] + ['left']
                                  ,height=20))
    fig.append_trace(trace5, 2, 1)


    fig.update_layout(height=650
                      ,width=800
                      ,margin = dict(l=70, r=50, t=50, b=10))


    fig.show()

# Обучение целевой модели    
def traning_model(data_train, data_test, model, RANDOM_SEED):
    
    start = datetime.now()

    X = data_train.drop(['default'], axis=1)
    y = data_train['default'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)

    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    
    period = datetime.now() - start
    print('Время выполнения: {}'.format(period))
    
    return  X_train, X_test, y_train, y_test, model, y_pred_prob, y_pred


# Поиск наилучших гиперпараметров
def revealing_best_parameters(reg_model, max_iter, tol_stop):
    
    start = datetime.now()

    param_grid = [
        {'penalty': ['l1'],
         'solver': ['liblinear', 'lbfgs'], 
         'class_weight': ['balanced'], 
         'multi_class': ['auto','ovr', 'ovo'], 
         'max_iter': [max_iter],
         'tol': [tol_stop]},
        {'penalty': ['l2'], 
         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
         'class_weight': ['balanced'], 
         'multi_class': ['auto','ovr', 'ovo'], 
         'max_iter': [max_iter],
         'tol': [tol_stop]},
        {'penalty': ['none'], 
         'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
         'class_weight': ['balanced'], 
         'multi_class': ['auto','ovr', 'ovo'], 
         'max_iter': [max_iter],
         'tol': [tol_stop]}
    ]
    gridsearch = GridSearchCV(reg_model, param_grid, scoring='f1', n_jobs=-1, cv=5)
    gridsearch.fit(X_train, y_train)
    reg_model = gridsearch.best_estimator_
    ##печатаем параметры
    best_parameters = reg_model.get_params()

    period = datetime.now() - start
    print('Время выполнения: {}\n'.format(period))
    print('Набор параметров для копирования:'.format(period))


    for i, parameter_name in enumerate(best_parameters.keys()):
        value = best_parameters[parameter_name]
        if isinstance(value, str):
            to_print = "\t\t\t\t\t\t\t\t ,{} = '{}'\n".format(parameter_name, best_parameters[parameter_name])
        else: 
            to_print = "\t\t\t\t\t\t\t\t ,{} = {}\n".format(parameter_name, best_parameters[parameter_name])
        if i == 0:
            to_print = to_print.replace(',','')
        print(to_print, end='')
        
 

def view_cross_validation(model, X_test, y_test, cv=10):
    
    start = datetime.now()
    
    temp_vec = cross_validate(model
                              ,X_test
                              ,y_test
                              ,cv=cv
                              ,scoring=('roc_auc')
                              ,verbose=1
                              ,return_train_score=True)
    
    x = np.arange(1,cv+1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=temp_vec['train_score']
                             ,mode='lines+markers'
                             ,name='тренировочные значения'
                             ,line = dict(shape = 'linear', color = 'rgb(10, 12, 240)')))
    fig.add_trace(go.Scatter(x=x, y=[np.mean(temp_vec['train_score']) for x in list(temp_vec['train_score'])]
                             ,mode='lines'
                             ,name='ср. тренировочное значение'
                             ,line = dict(shape = 'linear', color = 'rgb(10, 12, 240)', dash = 'dot')))
    fig.add_trace(go.Scatter(x=x, y=temp_vec['test_score']
                             ,mode='lines+markers'
                             ,name='тестовые значения'
                             ,line = dict(shape = 'linear', color = 'rgb(100, 10, 100)')))
    fig.add_trace(go.Scatter(x=x, y=[np.mean(temp_vec['test_score']) for x in list(temp_vec['test_score'])]
                             ,mode='lines'
                             ,name='ср. значение на тесте'
                             ,line = dict(shape = 'linear', color = 'rgb(100, 10, 100)', dash = 'dot')))

    fig.update_layout(width=800
                      ,height=350
                      ,title = 'Кросс-валидация по метрике ROC-AUC',
                         title_x = 0.5,
                         xaxis_title = 'Номер фолда',
                         yaxis_title = 'ROD-AUC',
                         legend = dict(x = .8, y = 0.05,xanchor = 'center', orientation = 'v'),
                         #barmode = 'overlay',
                         margin = dict(l=50, r=50, t=50, b=20))


    fig.show() 
    period = datetime.now() - start
    print('Время выполнения: {}\n'.format(period))
    