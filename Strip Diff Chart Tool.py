import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from matplotlib import figure
import datetime
from datetime import datetime as dt
import dateutil.relativedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings

import pandas as pd
# from pandas.core.common import SettingWithCopyWarning

start_time = dt.now()

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def __datetime(date_str):
    return dt.strptime(date_str, '%Y-%m-%d')

def dividendDatesAdd():
    dividendDatesStock1.append(df1['Date'][df1.index[0]])
    dividendDatesStock2.append(df2['Date'][df2.index[0]])
    for x in ticker1.dividends.index:
        dividendDatesStock1.append(__datetime(str(x)[:10]))
    for y in ticker2.dividends.index:
        dividendDatesStock2.append(__datetime(str(y)[:10]))

def resetIndex(df):
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

def accrualDaysUpdate(df1,dividendDatesStock1):
    for ind in df1.index:
        cnt = -1

        while (df1['Date'][df1.index[-1]-ind] < dividendDatesStock1[cnt]):
            cnt = (cnt-1)%len(dividendDatesStock1)

        df1['Accrual Days'][df1.index[-1]-ind] = (df1['Date'][df1.index[-1]-ind]-dividendDatesStock1[cnt]).days

def accrualsTillNow(df,ticker):
    couponAmount = float(input(f"Enter Coupon Amount for {ticker.info['symbol']} : "))
    dividend = couponAmount/360
    for ind in df.index:
        df['Accrual'][ind] = round(df['Accrual Days'][ind]*dividend , 3)

def stripPrice(df):
    df['Strip Price'] = ''
    for idx in df.index:
        df['Strip Price'] = round(df['Close'].astype(float) - df['Accrual'].astype(float) , 2)



stock1 = input("Enter Stock 1 : ")
ticker1 = yf.Ticker(stock1)
stock2 = input("Enter Stock 2 : ")
ticker2 = yf.Ticker(stock2)

time_period = 12

yf.pdr_override()
data1 = yf.download(stock1)

data2 = yf.download(stock2)

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
resetIndex(df1)

resetIndex(df2)

start_date = df1['Date'][0] if df1['Date'].size < df2['Date'].size else df2['Date'][0]


# df1['Date'][0]

# start_date = __datetime(df1['Date'][0] if df1['Date'].size < df2['Date'].size else df2['Date'][0])


dividendDatesStock1 = []
dividendDatesStock2 = []

df1['Accrual Days'] = ''
df2['Accrual Days'] = ''

for idx in df1.index:
    df1['Date'][idx] = __datetime(df1['Date'][idx])

for idx in df2.index:
    df2['Date'][idx] = __datetime(df2['Date'][idx])


dividendDatesAdd()

def accrualDaysUpdate(df1,dividendDatesStock1):
    for ind in df1.index:
        cnt = -1

        while (df1['Date'][df1.index[-1]-ind] < dividendDatesStock1[cnt]):
            cnt = (cnt-1)%len(dividendDatesStock1)

        df1['Accrual Days'][df1.index[-1]-ind] = (df1['Date'][df1.index[-1]-ind]-dividendDatesStock1[cnt]).days

accrualDaysUpdate(df1,dividendDatesStock1)
accrualDaysUpdate(df2,dividendDatesStock2)

df1['Accrual'] = 0
df2['Accrual'] = 0
accrualsTillNow(df1,ticker1)
accrualsTillNow(df2,ticker2)


stripPrice(df1)
stripPrice(df2)


df_new = pd.DataFrame()

flag = True
temp = ''

if df1.size < df2.size:
    flag = False
    
flag

if flag :
    df_new['Date'] = df2['Date']
else :
    df_new['Date'] = df1['Date']

df_new['Strip Price'] = 0
if flag:
    i = 0
    while df_new['Date'][0] != df1['Date'][i]:
        i = i+1
    
    for j in range(i,df1['Strip Price'].size):

        df_new['Strip Price'][j-i] = df1['Strip Price'][j] - df2['Strip Price'][j-i]
else :
    i = 0
    while df_new['Date'][0] != df2['Date'][i]:
        i = i+1
    for j in range(i,df2['Strip Price'].size):
        df_new['Strip Price'][j-i] = df1['Strip Price'][j-i] - df2['Strip Price'][j]

for i in df_new.index:
    df_new['Date'][i] = df_new['Date'][i].date()

c_title = f"{ticker1.info['symbol']} - {ticker2.info['symbol']}"

c_area = px.line(x=df_new['Date'], y=df_new['Strip Price'], title=c_title)

c_area.update_xaxes(
    title_text = 'Date',
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
            dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
            dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
            dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
            dict(step = 'all')])))

c_area.update_yaxes(title_text = 'Price', tickprefix = '$')
c_area.update_layout(showlegend = False,
    title = {
        'text': f"{ticker1.info['symbol']} - {ticker2.info['symbol']}",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

c_area.show()

df_new['Diff + Accrual'] = ''
accrual_diff = df1['Accrual'][df1.index[-1]] - df2['Accrual'][df2.index[-1]]
accrual_diff


for idx in df_new.index:
    df_new['Diff + Accrual'][idx] = round(df_new['Strip Price'][idx] + accrual_diff, 2)

c_area = px.line(x=df_new['Date'], y=df_new['Diff + Accrual'], title="PSA.PRO - USB.PQ")

c_area.update_xaxes(
    title_text = 'Date',
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
            dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
            dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
            dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
            dict(step = 'all')])))

c_area.update_yaxes(title_text = 'Price', tickprefix = '$')
c_area.update_layout(showlegend = False,
    title = {
        'text': f"{ticker1.info['symbol']} - {ticker2.info['symbol']} with Accrual Difference",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

c_area.show()

# do your work here
end_time = dt.now()
print('Duration: {}'.format(end_time - start_time))
