# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:04:13 2021

@author: David Lloyd
"""
import pandas as pd
from prophet import Prophet
import datetime
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Housing_PPI_Q1_2015_onwards.csv',header = 'infer')
date_0 = pd.Timestamp(2015,1,1)
#dt_ = datetime.timedelta(days = int(np.round(365.25/4)))
date_col = [date_0 + datetime.timedelta(days = ii*int(np.round(365.25/4)))\
            for ii in range(0,len(df))]


dt_ = datetime.timedelta(weeks = 6)
df['ds'] = pd.date_range('2015-1-1', periods=25, freq='QS')  + dt_

df.rename(columns = {'PPI':'y'}, inplace = True)

#%%

df_plot = df.copy(deep = True)
df_plot.set_index('ds',inplace = True)
df_plot.rename(columns = {'y': 'Quarterly PPI'}, inplace = True)

df_plot.plot(marker = 's', markersize = 6,\
             linestyle = '', figsize = (8,6))

plt.ylabel('PPI (ref. 2015 avg.)',fontsize = 18)
plt.xlabel('Date',fontsize = 18)
#plt.legend('')
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

x_temp = np.arange(0,len(df_plot),1)
c_fit = np.polyfit(x_temp,df_plot['Quarterly PPI'],deg = 1)

plt.plot(df_plot.index, c_fit[1] + c_fit[0]*x_temp, label = 'Linear fit')
plt.legend(fontsize = 14)
print(c_fit)

plt.savefig('PPI_data.png', dpi = 200, \
            transparent = True, bbox_inches='tight')

#%%
import matplotlib.ticker as ticker

bar_df = df_plot.diff().iloc[1:]
ax = (bar_df).plot(kind = 'bar', color = 'purple',legend=None)
plt.ylabel('Quarterly change in PPI', fontsize = 18)
plt.xlabel('End date', fontsize = 18)
plt.xticks(ha = 'right')
ticklabels = ['']*len(bar_df.index)
# Every 4th ticklable shows the month and day
ticklabels[3::4] = [item.strftime('%Y')+'- Q1' \
                    for item in bar_df.index[3::4]]
# Every 12th ticklabel includes the year
#ticklabels[::12] = [item.strftime('%b %d\n%Y') for item in bar_df.index[::12]]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))#, rotation = 90)

#ticklabels_min = ['']*len(bar_df.index)
#ticklabels_min[1::4] = ['Q3' for item in bar_df.index[1::4]]
#ax.xaxis.set_minor_formatter(ticker.FixedFormatter(ticklabels_min))

#plt.legend()
plt.gcf().autofmt_xdate()

plt.savefig('PPI_diff_bar.png', dpi = 200, \
            transparent = True, bbox_inches='tight')

#%%
m = Prophet(interval_width=0.95, daily_seasonality= False,\
            weekly_seasonality= False,\
                seasonality_mode= 'multiplicative',\
                #n_changepoints = 0), \
                    changepoint_prior_scale=0.0001)
# changepoint prior scale default = 0.05
m.fit(df)

#%%
future = pd.date_range('2015-1-1', periods=33, freq='QS')  + dt_
future = pd.DataFrame(future, columns = ['ds'])
#m.make_future_dataframe(periods=2,freq = 'Q')

forecast = m.predict(future)

#%%

fig1 = m.plot(forecast,figsize = (8,6))
ax = fig1.gca()
ax.set_title("Forecasted House Price", fontsize=20)
ax.set_xlabel("Date", fontsize=18)
ax.set_ylabel("PPI", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),fontsize = 16)
ax.set_yticklabels(ax.get_yticklabels(),fontsize = 16)

plt.savefig('main_forecast.png', dpi = 200, \
            transparent = True, bbox_inches='tight')
#%% zoom in on the above
fig1 = m.plot(forecast,figsize = (8,6))
ax = fig1.gca()

ax.set_xlim(pd.Timestamp(2021,1,1),pd.Timestamp(2023,1,1))
ax.set_ylim(126,155)
#ax.set_title("Forecasted House Price", fontsize=20)
ax.set_xlabel("Date", fontsize=32)
ax.set_ylabel("PPI", fontsize=32)
plt.xticks(rotation = 45, ha = 'right', fontsize = 30)
plt.yticks(rotation = 45, ha = 'right', fontsize = 30)
#ax.set_yticklabels(ax.get_yticklabels(),fontsize = 14)

plt.savefig('forecast_zoom.png', dpi = 200, \
            transparent = False, bbox_inches='tight')

#%%
fig2 = m.plot_components(forecast)
ax2 = fig2.gca()
ax2.set_xlabel('Date', fontsize = 18)

#%%
season_df = m.predict_seasonal_components(forecast)
#m.plot_yearly(forecast)

#%% long term prediction

future2 = pd.date_range('2015-1-1', periods=53, freq='QS')  + dt_
future2 = pd.DataFrame(future2, columns = ['ds'])
#m.make_future_dataframe(periods=2,freq = 'Q')

forecast2 = m.predict(future2)

fig1 = m.plot(forecast2,figsize = (8,6))
ax = fig1.gca()
ax.set_title("Forecasted House Price", fontsize=20)
ax.set_xlabel("Date", fontsize=18)
ax.set_ylabel("PPI", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),fontsize = 16)
ax.set_yticklabels(ax.get_yticklabels(),fontsize = 16)

#%% performance metric

from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='1095 days', \
                         period='180 days', horizon = '730 days')
metric_df = performance_metrics(df_cv)
metric_df.head()