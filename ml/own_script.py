# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd

import matplotlib



df = pd.read_csv('~/advanced_practical_2/ml/gbp_eur_2.csv') #give path to csv file

df_1 = df[['Datum', 'Laatste']] #csv file contains more data then needed

df_1['Datum'] = pd.to_datetime(df_1['Datum'], format = '%d/%m/%Y') 
plot_whole_dataset= df_1.plot(x='Datum', y='Laatste')
fig_whole_dataset = plot_whole_dataset.get_figure()
fig_whole_dataset.savefig('./advanced_practical_2/ml/plot_whole_dataset.png')
kurto_whole_set = df_1['Laatste'].kurtosis()
skew_whole_set = df_1['Laatste'].skew()
mean_whole_set = df_1['Laatste'].mean()
std_whole_set = df_1['Laatste'].std()
autocorr_whole_set = df_1['Laatste'].autocorr()

##get data into the 6 stages
##make function which makes dynamical subsets based on input split dates
mask_stg_1 = (df_1['Datum'] >= '2016-05-23') & (df_1['Datum'] < '2016-06-23')
mask_stg_2 = (df_1['Datum'] >= '2016-06-23') & (df_1['Datum'] < '2018-11-13')
mask_stg_3 = (df_1['Datum'] >= '2018-11-13') & (df_1['Datum'] < '2019-01-14')
mask_stg_4 = (df_1['Datum'] >= '2019-01-14') & (df_1['Datum'] < '2019-03-12')
mask_stg_5 = (df_1['Datum'] >= '2019-03-12') & (df_1['Datum'] < '2019-03-29')
mask_stg_6 = (df_1['Datum'] >= '2019-03-29') & (df_1['Datum'] < '2019-05-24')

df_stg_1 = df_1.loc[mask_stg_1]
df_stg_2 = df_1.loc[mask_stg_2]
df_stg_3 = df_1.loc[mask_stg_3]
df_stg_4 = df_1.loc[mask_stg_4]
df_stg_5 = df_1.loc[mask_stg_5]
df_stg_6 = df_1.loc[mask_stg_6]
## get sample stats 
## rep work -> might make function for it 
kurto_stg_1 = df_stg_1['Laatste'].kurtosis()
skew_stg_1 = df_stg_1['Laatste'].skew()
mean_stg_1 = df_stg_1['Laatste'].mean()
std_stg_1 = df_stg_1['Laatste'].std()
autocorr_stg_1 = df_stg_1['Laatste'].autocorr()

kurto_stg_2 = df_stg_2['Laatste'].kurtosis()
skew_stg_2 = df_stg_2['Laatste'].skew()
mean_stg_2 = df_stg_2['Laatste'].mean()
std_stg_2 = df_stg_2['Laatste'].std()
autocorr_stg_2 = df_stg_2['Laatste'].autocorr()

kurto_stg_3 = df_stg_3['Laatste'].kurtosis()
skew_stg_3 = df_stg_3['Laatste'].skew()
mean_stg_3 = df_stg_3['Laatste'].mean()
std_stg_3 = df_stg_3['Laatste'].std()
autocorr_stg_3 = df_stg_3['Laatste'].autocorr()

kurto_stg_4 = df_stg_4['Laatste'].kurtosis()
skew_stg_4 = df_stg_4['Laatste'].skew()
mean_stg_4 = df_stg_4['Laatste'].mean()
std_stg_4 = df_stg_4['Laatste'].std()
autocorr_stg_4 = df_stg_4['Laatste'].autocorr()

kurto_stg_5 = df_stg_5['Laatste'].kurtosis()
skew_stg_5 = df_stg_5['Laatste'].skew()
mean_stg_5 = df_stg_5['Laatste'].mean()
std_stg_5 = df_stg_5['Laatste'].std()
autocorr_stg_5 = df_stg_5['Laatste'].autocorr()

kurto_stg_6 = df_stg_6['Laatste'].kurtosis()
skew_stg_6 = df_stg_6['Laatste'].skew()
mean_stg_6 = df_stg_6['Laatste'].mean()
std_stg_6 = df_stg_6['Laatste'].std()
autocorr_stg_6 = df_stg_6['Laatste'].autocorr()
###get plots of small datasets
plot_stg_1= df_stg_1.plot(x='Datum', y='Laatste')
fig_stg_1 = plot_whole_dataset.get_figure()
fig_stg_1.savefig('./advanced_practical_2/ml/plot_stg_1.png')

plot_stg_2= df_stg_2.plot(x='Datum', y='Laatste')
fig_stg_2 = plot_whole_dataset.get_figure()
fig_stg_2.savefig('./advanced_practical_2/ml/plot_stg_2.png')

plot_stg_3= df_stg_3.plot(x='Datum', y='Laatste')
fig_stg_3 = plot_whole_dataset.get_figure()
fig_stg_3.savefig('./advanced_practical_2/ml/plot_stg_3.png')

plot_stg_4= df_stg_4.plot(x='Datum', y='Laatste')
fig_stg_4 = plot_whole_dataset.get_figure()
fig_stg_4.savefig('./advanced_practical_2/ml/plot_stg_4.png')

plot_stg_5= df_stg_5.plot(x='Datum', y='Laatste')
fig_stg_5 = plot_whole_dataset.get_figure()
fig_stg_5.savefig('./advanced_practical_2/ml/plot_stg_5.png')

plot_stg_6= df_stg_6.plot(x='Datum', y='Laatste')
fig_stg_6 = plot_whole_dataset.get_figure()
fig_stg_6.savefig('./advanced_practical_2/ml/plot_stg_6.png') 


