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

def __datetime(date_str):
    return dt.strptime(date_str, '%Y-%m-%d')

def dividendDatesAdd():
    dividendDatesStock1.append(prevDivDateStock1)
    dividendDatesStock2.append(prevDivDateStock2)
    for x in ticker1.dividends.index:
        if x >= start_date:
            dividendDatesStock1.append(__datetime(str(x)[:10]))
    for y in ticker2.dividends.index:
        if y >= start_date:
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

time_period = int(input("Historical Data Time Period (in months) : "))
end_date = dt.today().date()

start_date = end_date - dateutil.relativedelta.relativedelta(months=time_period)

yf.pdr_override()
data1 = pdr.get_data_yahoo(stock1, start=str(start_date), end=str(end_date))

data2 = pdr.get_data_yahoo(stock2, start=str(start_date), end=str(end_date))

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

dividendDatesStock1 = []
dividendDatesStock2 = []




divDates1 = []
divDates2 = []

for x in ticker1.dividends.index:
    divDates1.append(__datetime(str(x)[:10]))

for x in ticker2.dividends.index:
    divDates2.append(__datetime(str(x)[:10]))
    

prevDivDateStock1 = ''
prevDivDateStock2 = ''
for i in range(len(divDates1)):
    if divDates1[i].date() > start_date:
        prevDivDateStock1 = divDates1[i-1]
        break
        
for i in range(len(divDates2)):
    if divDates2[i].date() > start_date:
        prevDivDateStock2 = divDates2[i-1]
        break



dividendDatesAdd()

resetIndex(df1)

resetIndex(df2)

df1['Accrual Days'] = ''
df2['Accrual Days'] = ''

for idx in df1.index:
    df1['Date'][idx] = __datetime(df1['Date'][idx])
    df2['Date'][idx] = __datetime(df2['Date'][idx])


accrualDaysUpdate(df1,dividendDatesStock1)
accrualDaysUpdate(df2,dividendDatesStock2)

df1['Accrual'] = ''
df2['Accrual'] = ''
accrualsTillNow(df1,ticker1)
accrualsTillNow(df2,ticker2)


stripPrice(df1)
stripPrice(df2)


df_new = pd.DataFrame()


df_new['Date'] = ''
df_new['Strip Price Diff'] = (df1['Strip Price']-df2['Strip Price'])


for idx in df1.index:
    df_new['Date'][idx] = df1['Date'][idx].date()


# f = plt.figure()
# f.set_figwidth(30)
# f.set_figheight(15)
# x = df_new['Date']
# y = df_new['Strip Price Diff']
# plt.plot(x,y)
# plt.show()

c_area = px.area(x=df_new['Date'], y=df_new['Strip Price Diff'], title="PSA.PRO - USB.PQ")

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
        'text': 'PSA.PRO - USB.PQ',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

c_area.show()

df_new['Diff + Accrual'] = ''
accrual_diff = df1['Accrual'][df1.index[-1]] - df2['Accrual'][df2.index[-1]]
accrual_diff


for idx in df_new.index:
    df_new['Diff + Accrual'][idx] = df_new['Strip Price Diff'][idx] + accrual_diff

# f = plt.figure()
# f.set_figwidth(30)
# f.set_figheight(15)
# x = df_new['Date']
# y = df_new['Diff + Accrual']
# plt.plot(x,y)
# plt.show()

c_area = px.area(x=df_new['Date'], y=df_new['Diff + Accrual'], title="PSA.PRO - USB.PQ")

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
        'text': 'PSA.PRO - USB.PQ',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

c_area.show()


