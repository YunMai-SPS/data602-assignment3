# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:03:57 2017

@author: yun
"""

# get the date and time for blotter and PL table
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup # web scraping
import requests # package for retrieving HTML
from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine

import MySQLdb
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table

import pandas_datareader as pdr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ochl,candlestick_ohlc

from datetime import datetime, timedelta

from io import  StringIO, BytesIO
import base64
import urllib.parse

import matplotlib.dates as mdates
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.gridspec as gridspec

import matplotlib.ticker as mticker
import matplotlib
from pylab import rcParams

import urllib.request
import seaborn as sns
from scipy import stats

import talib  # stock indicator calculation

import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

# read the equity list
url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'
ticker_list = pd.read_csv(url, sep=',', skipinitialspace=True, header=None, engine='python')
ticker_list.drop(ticker_list.columns[[9]], axis=1,inplace=True)
ticker_list.columns = [ticker_list.iloc[0]]
ticker_list.drop(ticker_list.index[0],inplace=True)

# convert the dataframe to dictionary for the html selection form
s1 = pd.Series(list(['name'] * len(ticker_list)), index=list(range(1,len(ticker_list)+1)),name='key')
s2 = pd.Series(ticker_list['Symbol'])
pre_dic = pd.concat([s1, s2], axis=1)
pre_dic = pre_dic.sort_values('Symbol')

dic = list()
for i in range(0,len(ticker_list)):
    x=pre_dic.iloc[i].to_frame().transpose()
    dic1 = x.set_index('key')['Symbol'].to_dict()
    dic.append(dic1) 
      
#create the blotter in dataframe
blotter = pd.DataFrame({'Side' : "",
                        'Ticker' : '',
                        'Quantity' : 0.,
                        'Executed Price' : 0.,
                        'Date / Time' : time.strftime("%Y-%m-%d")+'/'+time.strftime("%H:%M:%S"),
                        'Cash': 10000000.00,
                        'Cost' : 0.},index=[0],columns=['Side', 'Ticker', 'Quantity', 'Executed Price', 'Date/Time', 'Cost','Cash'])

#create the PL table in dataframe
pl = pd.DataFrame({'Ticker' : list(s2),
                   'Position' : 0.,
                   'Market': 0.,
                   'RPL' : 0.,
                   'UPL' : 0.,
                   'Total P/L': 0.,
                   'Allocation By Shares' : 0.,
                   'Allocation By Dollars' : 0.,
                   'WAP': 0.,
                   'Correlation-Weather': 0.},columns=['Ticker', 'Position', 'Market', 'WAP', 'RPL','UPL','Total P/L','Allocation By Shares','Allocation By Dollars','Correlation-Weather'])

# function for extract number incase the user input invalid string
def ext_num(x):
    return float(''.join(y for y in x if y.isdigit() or y == '.'))
    
#function for getting the market price of stock
def price(ticker): 
    # retrieves the Yahoo Finance HTML
    url = "http://www.nasdaq.com/symbol/"+ticker+"/real-time"
    page = requests.get(url)
    page.status_code # see whetherthe  page has been downloaded successfully
    # parsing the page with BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')
    # fetch the current price for the selected equity
    price_per_share = soup.find(id='quotes_content_left__LastSale').get_text()
    price_per_share = float(price_per_share.replace(',',''))
    return price_per_share

# function for calculate total dollar expended or earned in on trasaction
def trade_cost(p,v):
    total =  float(p) * float(v)
    return total

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/trade_interface/',methods = ['GET','POST'])
def select_trade():
    return redirect(url_for("trade_start"))

@app.route('/trade1_interface/start')
def trade_start():
    return render_template('form_select4_multi_action_for3.html',data=dic)

@app.route('/stock?')
def pre_trade():
    return render_template('form_select4_c.html',data=dic)

@app.route("/last100_trades", methods=['GET', 'POST'])
def last100():
    errors = []
    select = request.form.get('ticker_select')
    select = str(select).lower()
    sldic = {select.upper()}
    plot_data = str()
    sl1 = pd.Series('name', name='key')
    sl2 = pd.Series(str(select).upper(),name='Symbol')
    pre_dic2 = pd.concat([sl1, sl2], axis=1)
    dic2 = pre_dic2.set_index('key')['Symbol'].to_dict()
    if request.method == "POST":
        # get url that the user has entered
        try:
            market = pd.DataFrame()
            market_sr = pd.Series()
            for i in range(1,3):
                url_1 = "http://www.nasdaq.com/symbol/"+select+"/time-sales?time=0&pageno="+str(i)
                page = requests.get(url_1)
                soup = BeautifulSoup(page.content,'html.parser')
                afterhour = soup.find('table', id="AfterHoursPagingContents_Table")
                market1 = afterhour.select('tr')
                market_raw = [mk.get_text() for mk in market1]
                market_raw = pd.Series(market_raw)
                market_sr = market_sr.append(market_raw)
            market = market_sr.str.split('\n',expand=True)
            cols = [0,4]
            market.drop(market.columns[cols],axis=1,inplace=True)
            market.columns = market.iloc[0]
            market.drop(market.index[0],inplace=True)
            market['NLS Time (ET)'] = market['NLS Time (ET)'].str.strip(to_strip=None)
            market.index=market.reset_index().index
            
            market['NLS Time (ET)'] = market['NLS Time (ET)'].str.strip(to_strip=None)
            add_date = datetime.now().date().strftime('%Y-%m-%d')+" "
            mk_time = list(add_date + market['NLS Time (ET)'].astype(str))
            x = [datetime.strptime(d,'%Y-%m-%d %H:%M:%S') for d in mk_time]
            y = market['NLS Price'].str.replace(u'\xa0', u'').replace( {'\$': '', ',': ''}, regex=True ).astype(float)
            vol = market['NLS Share Volume'].replace( '[\,]','', regex=True ).astype(float)
             
            
            # draw figure
            img = BytesIO()  # create the buffer
            rcParams['figure.figsize'] = 10, 7
            plt.style.use('seaborn-ticks')
                
            fig = plt.figure(facecolor='white')
        
            ax1 = plt.subplot2grid((5,4),(0,0),rowspan=4,colspan=4)
            ax1.plot(x,y, color='#17becf')
            plt.ylabel('Stock Price')
            # color settings
            ax1.set_facecolor("white")
    
            ax2 = plt.subplot2grid((5,4),(4,0),sharex=ax1,rowspan=4,colspan=4)
            ax2.bar(x,vol,color='#2ca02c',width=0.0001,align='center')
            plt.ylabel('Volume')
            
            # color settings
            ax2.set_facecolor("white")
           
            # hide x axis tick label for ax1
            plt.setp(ax1.get_xticklabels(),visible=False)
            
            plt.subplots_adjust(left=.10,bottom=.20,right=.95,top=0.92,wspace=.20,hspace=.005)
            
            plt.savefig(img, format='png',facecolor=fig.get_facecolor(), edgecolor='#07000d')  # save figure to the buffer
            img.seek(0)  # rewind the buffer
    
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
            
        except:
            errors.append(
                "Unable to get market price. Please make sure the ticker symbol is valid and try again."
            )
                
    return render_template('form_input_image_2.html', plot_url=plot_data, selectdic=sldic)

def HisDataGain(url):
    with requests.Session() as session:
        # parsing parameters
        response = session.get(url)
        soup = BeautifulSoup(response.content,"lxml")
    
        data = {
            'ctl00$quotes_content_left$submitString': '1y|false|AMZN',
            '__VIEWSTATE': soup.find('input', {'name': '__VIEWSTATE'}).get('value', ''),
            '__VIEWSTATEGENERATOR': soup.find('input', {'name': '__VIEWSTATEGENERATOR'}).get('value', ''),
            '__EVENTVALIDATION': soup.find('input', {'name': '__EVENTVALIDATION'}).get('value', ''),
            '__VIEWSTATEENCRYPTED': soup.find('input', {'name': '__VIEWSTATEENCRYPTED'}).get('value', '')
        }
    
        # parsing data
        response = session.post(url, data=data)
    
        soup = BeautifulSoup(response.content, "lxml")
    
        div = soup.find(id="quotes_content_left_pnlAJAX")
        table = div.find("table")
        data=[]
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            cell = [ele.text.strip() for ele in cols]
            data.append(cell)
        
        header = table.find_all("tr")[0]
        while header.find("span"):
            header.find("span").extract()
        cols = header.find_all("th")
        headerName = [ele.text.strip() for ele in cols]
    
    history1year = pd.DataFrame(np.array(data[2:]).reshape(len(data)-2,6),columns=headerName)
    history1year[['Open', 'High', 'Low', 'Close / Last']] = history1year[['Open', 'High', 'Low', 'Close / Last']].astype(float)
    history1year[['Volume']] = history1year.iloc[:,-1].str.replace(",","").astype(float)
    history1year[['Date']] = pd.to_datetime(history1year.iloc[:,0])
    return history1year

def WeatherDataGain(url):
    # url_2 = 'https://raw.githubusercontent.com/YunMai-SPS/data602_data/master/icao_world.csv'
    #the comment is the code if the weather of other cities are consider
    #airport_data = pd.read_csv(url_2, delimiter='\t', encoding = "ISO-8859-1", engine='python')
    
    #input_city = 'New York'
    #location = airport_data[airport_data.city == input_city][['ICAO']]
    #location = str(location['ICAO'].tolist()[0])
    page = requests.get(url)
    soup = BeautifulSoup(page.content,'html.parser')
    weatherHeader = soup.find('table', id='obsTable')
    weatherCol_raw = weatherHeader.select('th')
    weatherCol = [wth.get_text() for wth in weatherCol_raw]
    weatherCol_Ar = np.reshape(weatherCol,(-1,9))
    weatherCol = pd.DataFrame(weatherCol_Ar)
    
    weatherCol = pd.concat([weatherCol[0], 
                            weatherCol[1],weatherCol[1],weatherCol[1],
                            weatherCol[2],weatherCol[2],weatherCol[2],
                            weatherCol[3],weatherCol[3],weatherCol[3],
                            weatherCol[4],weatherCol[4],weatherCol[4],
                            weatherCol[5],weatherCol[5],weatherCol[5],
                            weatherCol[6],weatherCol[6],weatherCol[6],
                            weatherCol[7],weatherCol[8] 
                            ], axis=1, join_axes=[weatherCol[0].index])
    weatherCol.columns = np.arange(21)
    
    weatherContent_raw = weatherHeader.select('td')
    weatherContent = [wth.get_text() for wth in weatherContent_raw]
    weatherContent = list(map(lambda x: x.strip(),weatherContent))
    weatherContent = list(map(lambda x: x.translate( { ord(c):None for c in ' \n\t\r' } ),weatherContent)) 
    weatherContent = np.array(weatherContent)
    weather_Ar = np.reshape(weatherContent, (-1,21))   
    weather = pd.DataFrame(weather_Ar)
       
    cols = weather.iloc[0]
    cols[0] = ''
    weatherCol.iloc[0][0] = ''
    weather.drop(weather[weather[1]=='high'].index,inplace=True)
    weather.columns =  weatherCol.iloc[0]+' '+cols  
        
    start_day = (datetime.now() - timedelta(days=1*365)).strftime("%Y%m%d")
    pd.date_range(start_day, periods=366)
    weather['Date'] = pd.date_range(start_day, periods=366)
    weather.drop(weather.columns[[0]], axis=1,inplace=True)
    return weather        

def Rsquared(df):
    r_2 = list()
    vars = ['Temp. (°F) avg','Dew Point (°F) avg', 'Humidity (%) avg', 'Sea Level Press. (in) avg',
            'Visibility (mi) avg', 'Wind (mph) avg','Precip. (in) sum']
    for var in vars:
        x=df[var]
        y=df['Close / Last']
        mask = ~np.isnan(x) & ~np.isnan(y)
        stat1 = stats.linregress(x[mask],y[mask])
        r_2.append(stat1[2])
    return r_2
    
@app.route("/trade/process", methods=['GET', 'POST'])
def trade():
    # set globle variables
    errors = []
    global blotter
    global pl
    
    select = 'tqqq'
    side = 'buy'
    volume = '100'
    volume = ext_num(volume)
    
    #add the correlation between stock price and weather condistion(temperatue) as a PL column
    '''
    get 1 year of historical data for the selected stock
    '''
    select = request.form.get('input_symbol')
    stock_lo = str(select).lower()
    sldic = {select.upper()}
    url = 'http://www.nasdaq.com/symbol/'+stock_lo+'/historical/'
    history1year = HisDataGain(url)
    
    '''
    get time-series data of NYC weather
    '''  
    location = 'KNYC'
    
    timedelta(days=1*365)
    oneYearAgo=(datetime.now() - timedelta(days=1*365)).strftime("%Y/%m/%d")
    
    currentDay = str(datetime.now().day)
    currentMonth = str(datetime.now().month)
    currentYear = str(datetime.now().year)
    
    url_3 = 'https://www.wunderground.com/history/airport/'+location+'/'+oneYearAgo+'/CustomHistory.html?dayend='+currentDay+'&monthend='+currentMonth+'&yearend='+currentYear+'&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo='
    
    weather = WeatherDataGain(url_3)
    
    '''
    combine the stock price data and the weather data
    '''    
    history_weather = weather.merge(history1year, how='left', left_on='Date', right_on='Date')
    history_weather.dropna(inplace=True)
    history_weather.columns.values[15] = 'Wind (mph) avg'
    history_weather.columns.values[16] = 'Wind (mph) low'
    cols = ['Temp. (°F) high', 'Temp. (°F) avg', 'Temp. (°F) low',
       'Dew Point (°F) high', 'Dew Point (°F) avg', 'Dew Point (°F) low',
       'Humidity (%) high', 'Humidity (%) avg', 'Humidity (%) low',
       'Sea Level Press. (in) high', 'Sea Level Press. (in) avg',
       'Sea Level Press. (in) low', 'Visibility (mi) high',
       'Visibility (mi) avg', 'Visibility (mi) low', 'Wind (mph) avg',
       'Wind (mph) low', 'Wind (mph) high', 'Precip. (in) sum' ]
   
    history_weather[cols] = history_weather[cols].replace({'-': np.nan,'T': np.nan}, regex=True)
    history_weather[cols] = history_weather[cols].astype(float) 
    
    '''
    calculate R squared of the stock price and weather correlation for adding to P/L
    '''
    r2 = Rsquared(history_weather)
    
    '''
    trade module
    '''
    #create the PL table in dataframe
    pl = pd.DataFrame({'Ticker' : list(s2),
                       'Position' : 0.,
                       'Market': 0.,
                       'RPL' : 0.,
                       'UPL' : 0.,
                       'Total P/L': 0.,
                       'Allocation By Shares' : 0.,
                       'Allocation By Dollars' : 0.,
                       'WAP': 0.,
                       'Correlation-Weather': 0.},columns=['Ticker', 'Position', 'Market', 'WAP', 'RPL','UPL','Total P/L','Allocation By Shares','Allocation By Dollars','Correlation-Weather'])
    output1 = pd.DataFrame()
    output2 = pd.DataFrame()
    current_price = 0.0
    volume = 0.0
    check_p = 0.0
    # select equity for the trade
    select = request.form.get('input_symbol')
    stock = str(select).upper()
    # select either buy or sell
    side = request.values.get('side')
    
    # get blotter and pl data from database(here I use AWS RDS MySQL engine)
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306')
    engine.execute("CREATE DATABASE IF NOT EXISTS trade_data") #create db
    engine.execute("USE trade_data") # select new db
    
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306/trade_data')
    conn = engine.connect()
    try:
        blotter = pd.read_sql('blotter', conn)
    except:
        blotter = pd.read_sql('blotter', conn)
    conn.close()
    blotter = blotter.set_index('index')
    
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306/trade_data')
    conn = engine.connect()
    try:
        pl_0 = pd.read_sql('pl', conn)
    except:
        pl_0 = pd.read_sql('pl', conn)
    conn.close()
   
    pl_0 = pl_0.set_index('index')
    col_to_fill = ['Position', 'Market', 'WAP', 'RPL', 'UPL', 'Total P/L',
       'Allocation By Shares', 'Allocation By Dollars', 'Correlation-Weather']
    pl.loc[pl.Ticker.isin(pl_0.Ticker), col_to_fill]=pl_0[col_to_fill].values
            
    # select equity for the trade
    if side == 'sell':
        check_p = pl.loc[pl['Ticker'] == stock,'Position'].iloc[0,]
        if check_p == 0.0:
            errors.append(
                "Sorry. You don't have stock to sell."
            )  
            pass
        else:
            # select the volume for the trade
            volume = request.values.get('equity_volume')
            volume = ext_num(volume)
            if volume > check_p:
                errors.append(
                "Sorry. The volume you put is out of bounds. Please input a new volume. Please go back to adjust the volume that you want to trade:"
                )         
                pass
            else:
                # update the Blotter 
                blotter.at[len(blotter),'Date/Time'] = time.strftime("%Y-%m-%d")+'/'+time.strftime("%H:%M:%S")
                blotter.at[(len(blotter)-1),'Side'] = side
                blotter.at[(len(blotter)-1),'Ticker'] = stock
                blotter.at[(len(blotter)-1),'Quantity'] = volume
                current_price = price(stock)
                blotter.at[(len(blotter)-1),'Executed Price'] = current_price 
                blotter.at[(len(blotter)-1),'Cost'] =  trade_cost(current_price,volume)
                spend = trade_cost(current_price,volume)  
                if blotter.shape[0] > 1:
                    blotter.at[(len(blotter)-1),'Cash'] =  blotter.at[(len(blotter)-2),'Cash'] + spend
                else:
                    blotter.at[(len(blotter)-1),'Cash'] =  blotter.at[(len(blotter)-1),'Cash'] + spend
                blotter = blotter.sort_values(by='Date/Time')
                blotter = blotter.iloc[::-1]
                
                # update Market, RPL, postition/inventory, UPL, Total P/L, Allocation By Shares, and Allocation By Dollars sequentially in the P/L table
                # Market
                pl.loc[pl['Ticker']==stock,'Market'] = current_price
                # RPL                            
                previous_rpl = pl.loc[pl['Ticker']==stock,'RPL'].iloc[0] 
                price_dif = current_price - pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                pl.loc[pl['Ticker']==stock,'RPL'] =  previous_rpl + volume * price_dif
                # position
                previous_position = pl.loc[pl['Ticker']==stock,'Position'].iloc[0]
                current_position = previous_position - volume
                pl.loc[pl['Ticker']==stock,'Position'] = current_position
                # if there is price change, update the UPL
                current_wap = pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                # When price is different from WAP or all stocks are sold, UPL need to be updated
                if current_price != float(format(current_wap, '.2f')) or current_position == 0.0:
                    pl.loc[pl['Ticker']==stock,'UPL'] =  price_dif * current_position  
                pl.loc[pl['Ticker']==stock,'Total P/L'] = pl.loc[pl['Ticker']==stock,'RPL'] + pl.loc[pl['Ticker']==stock,'UPL']
                total_share = pl.Position.sum()
                pl.loc[pl['Ticker']==stock,'Allocation By Shares'] = pl.loc[pl['Ticker']==stock,'Position']/total_share
                total_dollor = (pl.Position.sum()*pl.Market).sum()
                pl.loc[pl['Ticker']==stock,'Allocation By Dollars'] = (pl.loc[pl['Ticker']==stock,'Position']*pl.loc[pl['Ticker']==stock,'Market'])/total_dollor
                pl.loc[pl['Ticker']==stock,'Correlation-Weather'] = round(float(r2[0]),4)
                                
        #remove the bad log 
        if blotter.isnull().values.any():
            pl = pl_0
            blotter = blotter.dropna()
         
        output1 = blotter
        output2 = pl[pl.Market != 0]
        if (output2.shape[0] >1 and output2.shape[0] <2):
            re_cal_FirstTrade = blotter.loc[0].Ticker
            pl.loc[pl['Ticker']==re_cal_FirstTrade,'Allocation By Shares'] = pl.loc[pl['Ticker']==re_cal_FirstTrade,'Position']/total_share
            pl.loc[pl['Ticker']==re_cal_FirstTrade,'Allocation By Dollars'] = (pl.loc[pl['Ticker']==re_cal_FirstTrade,'Position']*pl.loc[pl['Ticker']==re_cal_FirstTrade,'Market'])/total_dollor
            output2 = pl[pl.Market != 0]
                                          
    if side == 'buy':
        # check there is cash in the account
        check_c = blotter.Cash
        if check_c[0] == 1.0:
            errors.append(
                "Cash is 0. Please replenish your account before trading."
            )  
            pass
        else:
            stock = str(select).upper()
             # select the volume for the trade
            volume = request.values.get('equity_volume')
            volume = ext_num(volume)
            current_price = price(stock)
            # make sure there is enough cash for the trasaction
            spend = trade_cost(current_price,volume)
            if spend > check_c[0]:
                errors.append(
                "There is not enough cash in your account. Please replenish before trading."
                )         
                pass
            else: 
                # update the Blotter
                # for the first buying in the day, update the first row
                if blotter.loc[0,'Quantity'] == 0.0:
                    blotter.at[(len(blotter)-1),'Date/Time'] = time.strftime("%Y-%m-%d")+'/'+time.strftime("%H:%M:%S")
                    blotter.at[(len(blotter)-1),'Side'] = side
                    blotter.at[(len(blotter)-1),'Ticker'] = stock
                    blotter.at[(len(blotter)-1),'Quantity'] = volume
                    blotter.at[(len(blotter)-1),'Executed Price'] = current_price 
                    blotter.at[(len(blotter)-1),'Cost'] = spend
                    spend = trade_cost(current_price,volume)  
                    blotter.at[(len(blotter)-1),'Cash'] =  blotter.at[(len(blotter)-1),'Cash'] - spend
                    blotter = blotter.sort_values(by='Date/Time')
                    blotter = blotter.iloc[::-1]
                else: # for the buyings thereafter, record a transaction in a new row
                    blotter.at[len(blotter),'Date/Time'] = time.strftime("%Y-%m-%d")+'/'+time.strftime("%H:%M:%S")
                    blotter.at[(len(blotter)-1),'Side'] = side
                    blotter.at[(len(blotter)-1),'Ticker'] = stock
                    blotter.at[(len(blotter)-1),'Quantity'] = volume
                    blotter.at[(len(blotter)-1),'Executed Price'] = current_price 
                    blotter.at[(len(blotter)-1),'Cost'] =  trade_cost(current_price,volume)
                    spend = trade_cost(current_price,volume)
                    if blotter.shape[0] > 1:
                        blotter.at[(len(blotter)-1),'Cash'] =  blotter.at[(len(blotter)-2),'Cash'] - spend
                    else:
                        blotter.at[(len(blotter)-1),'Cash'] =  blotter.at[(len(blotter)-1),'Cash'] - spend
                    blotter = blotter.sort_values(by='Date/Time')
                    blotter = blotter.iloc[::-1]
                # update Market, WAP, postition/inventory, UPL, Total P/L, Allocation By Shares, and Allocation By Dollars sequentially in the P/L table
                # Market               
                pl.loc[pl['Ticker']==stock,'Market'] = current_price
                 # put symbol if there is no record in P/L table
                
                # if there are more than one buyings, recalculate the WAP               
                if len(blotter.loc[(blotter['Ticker']==stock)&(blotter['Side'] == side)]) > 1:
                    new_cost = blotter.loc[(len(blotter)-1),'Cost'] 
                    previous_cost = pl.loc[pl['Ticker']==stock,'Position'].iloc[0] * pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                    total_volume = pl.loc[pl['Ticker']==stock,'Position'].iloc[0] + volume
                    pl.loc[pl['Ticker']==stock,'WAP'] = (new_cost + previous_cost)/total_volume
                    previous_position = pl.loc[pl['Ticker']==stock,'Position'].iloc[0]
                    current_position = previous_position + volume
                    pl.loc[pl['Ticker'].str.contains(stock),'Position'] = current_position
                    # if there is price change, update the UPL 
                    current_wap = pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                    price_dif = current_price - pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                    if current_price != float(format(current_wap, '.2f')):
                        pl.loc[pl['Ticker']==stock,'UPL'] =  price_dif * current_position
                    pl.loc[pl['Ticker']==stock,'Total P/L'] = pl.loc[pl['Ticker']==stock,'RPL'] + pl.loc[pl['Ticker']==stock,'UPL']
                    total_share = pl.Position.sum()
                    pl.loc[pl['Ticker']==stock,'Allocation By Shares'] = pl.loc[pl['Ticker']==stock,'Position']/total_share
                    total_dollor = (pl.Position*pl.Market).sum()
                    pl.loc[pl['Ticker']==stock,'Allocation By Dollars'] = (pl.loc[pl['Ticker']==stock,'Position']*pl.loc[pl['Ticker']==stock,'Market'])/total_dollor
                    pl.loc[pl['Ticker']==stock,'Correlation-Weather'] = round(float(r2[0]),4)
                else:
                    pl.loc[pl['Ticker']==stock,'WAP'] = current_price
                    # postition                     
                    previous_position = pl.loc[pl['Ticker']==stock,'Position'].iloc[0]
                    current_position = previous_position + volume
                    pl.loc[pl['Ticker'].str.contains(stock),'Position'] = current_position
                    # if there is price change, update the UPL 
                    current_wap = pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                    price_dif = current_price - pl.loc[pl['Ticker']==stock,'WAP'].iloc[0]
                    if current_price != float(format(current_wap, '.2f')):
                        pl.loc[pl['Ticker']==stock,'UPL'] =  price_dif * current_position
                    pl.loc[pl['Ticker']==stock,'Total P/L'] = pl.loc[pl['Ticker']==stock,'RPL'] + pl.loc[pl['Ticker']==stock,'UPL']
                    total_share = pl.Position.sum()
                    pl.loc[pl['Ticker']==stock,'Allocation By Shares'] = pl.loc[pl['Ticker']==stock,'Position']/total_share
                    total_dollor = (pl.Position*pl.Market).sum()
                    pl.loc[pl['Ticker']==stock,'Allocation By Dollars'] = (pl.loc[pl['Ticker']==stock,'Position']*pl.loc[pl['Ticker']==stock,'Market'])/total_dollor
                    pl.loc[pl['Ticker']==stock,'Correlation-Weather'] = round(float(r2[0]),4)
                    
        #remove the bad log 
        if blotter.isnull().values.any():
            pl = pl_0
            blotter = blotter.dropna()
            
        output1 = blotter
        output2 = pl[pl.Market != 0]
        
        if (output2.shape[0] >1 and output2.shape[0] <2):
            re_cal_FirstTrade = blotter.loc[0].Ticker
            pl.loc[pl['Ticker']==re_cal_FirstTrade,'Allocation By Shares'] = pl.loc[pl['Ticker']==re_cal_FirstTrade,'Position']/total_share
            pl.loc[pl['Ticker']==re_cal_FirstTrade,'Allocation By Dollars'] = (pl.loc[pl['Ticker']==re_cal_FirstTrade,'Position']*pl.loc[pl['Ticker']==re_cal_FirstTrade,'Market'])/total_dollor
            output2 = pl[pl.Market != 0]
            
    if side == 'do nothing':
        output1 = blotter
        output2 = pl[pl.Market != 0]
        
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306/trade_data')
    output1.to_sql('blotter', engine, if_exists='replace')
    output2.to_sql('pl', engine, if_exists='replace')
        
    #read the updated tables to display on the web page
    conn = engine.connect()
    try:
        showblt = pd.read_sql('blotter', conn)
    except:
        showblt = pd.read_sql('blotter', conn)
    conn.close()
    showblt = showblt.set_index('index')
    
    conn = engine.connect()
    try:
        showpl = pd.read_sql('pl', conn)
    except:
        showpl = pd.read_sql('pl', conn)
    conn.close()
    showpl = showpl.set_index('index')
    
    return render_template('view.html',tables=[showblt.to_html(classes='blt'),showpl.to_html(classes='pl')], 
        titles = ['na','Blotter','P/L']) 

# simple moving average
def SMA(price,period):
    weights = np.ones(period)/period
    sma = np.convolve(price,weights,'valid')
    return sma

# Moving average convergence divergence (MACD)
# MACD line = EMA_12- EMA_26
# single line = EMA_9 of MACD line
# histogram = MACD line - single line
def MACD(df, n_short, n_long):  
    EMAlong = pd.Series(pd.ewma(df['Close / Last'], span = n_short, min_periods = n_long - 1))  
    EMAshort = pd.Series(pd.ewma(df['Close / Last'], span = n_long, min_periods = n_long - 1))  
    MACD = pd.Series(EMAshort - EMAlong, name = 'MACD_' + str(n_short) + '_' + str(n_long))  
    MACDsigl = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACD_single_' + str(n_long) + '_' + str(n_short))  
    MACDhisg = pd.Series(MACD - MACDsigl, name = 'MACD_histograme_' + str(n_short) + '_' + str(n_long))    
    return MACD, MACDsigl, MACDhisg     


   
@app.route("/studies", methods=['GET', 'POST'])
def studies():
    studies_table = pd.DataFrame()
    plot_data = str()
    select = request.form.get('ticker_select')
    stock = str(select).lower()
    sldic = {select.upper()}
    url = 'http://www.nasdaq.com/symbol/'+stock+'/historical/'
    history1year = HisDataGain(url)
    
    stockfile = history1year.values.T.tolist()
    date = mdates.date2num(stockfile[0])   
    openp = np.loadtxt(stockfile[1], delimiter=',', unpack=True)
    highp = np.loadtxt(stockfile[2], delimiter=',',  unpack=True)
    lowp = np.loadtxt(stockfile[3], delimiter=',', unpack=True)
    closep = np.loadtxt(stockfile[4], delimiter=',', unpack=True)
    volume = np.loadtxt(stockfile[5], delimiter=',', unpack=True) 

    # prepare array for the condlestick chart: ochl
    
    candleAr = []
    for i in range(len(date)):
        appendLine = date[i],openp[i],closep[i],highp[i],lowp[i],volume[i]
        candleAr.append(appendLine)
        
    # prepare array for the condlestick chart: ohlc
    his_copy =  history1year.copy()
    his_copy.Date = date
    his1Y_Ar = [tuple(x) for x in his_copy[['Date', 'Open', 'High', 'Low', 'Close / Last', 'Volume']].to_records(index=False)]
    
    #global mean for closeing price
    glMean = np.repeat(history1year['Close / Last'].mean(),len(date))
    
    # 5-day and 20-day moving average
    mav5 = SMA(closep,5)
    mav20 = SMA(closep,20)
    
    start_5 = len(date[5-1:])
    start_20 = len(date[20-1:])
    
    # 5-day and 20-day moving standard deviation
    closep_5 = closep[np.arange(closep.size - 5 + 1)[:,None] + np.arange(5)]
    sd_5 = np.std(closep_5,axis=1)
    
    closep_20 = closep[np.arange(closep.size - 20 + 1)[:,None] + np.arange(20)]
    sd_20 = np.std(closep_20,axis=1)
    
    # daily price difference and daily return: (today close - yesterday close)/yesterday close
    dly_diff = np.diff(closep)
    dly_return = dly_diff/closep[:-1]
    dr_mean = np.repeat(dly_return.mean(),len(date)-1)
    dly_return_p = pd.Series(dly_return).apply(lambda x: '{:.4%}'.format(x)).values
    
    #Bollinger bands 2 standard deviation
    #Middle Band = 20-day simple moving average (SMA)
    #Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
    #Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
    up_band = mav20 + sd_20 *2 
    low_band = mav20 - sd_20 *2 
    
    #Price range for the day (bars)
    intraday_PriceRange = closep- openp
    
    #Daily price difference (high - low)
    intraday_PriceDiff = highp - lowp
    
    # MACD
    macd = MACD(history1year, 12, 26)
    
    # combine the statistics into one dataframe
    # use 0 in analysis_Ar to make sure there is no 0
    mav5_fill= np.insert(mav5, np.repeat(0,len(closep)-len(mav5)),np.zeros(len(closep)-len(mav5)) + np.nan)
    mav20_fill= np.insert(mav20, np.repeat(0,len(closep)-len(mav20)), np.zeros(len(closep)-len(mav20)) + np.nan)
    sd_5_fill= np.insert(sd_5, np.repeat(0,len(closep)-len(sd_5)),np.zeros(len(closep)-len(sd_5)) + np.nan)
    sd_20_fill= np.insert(sd_20, np.repeat(0,len(closep)-len(sd_20)),np.zeros(len(closep)-len(sd_20)) + np.nan)
    dly_diff_fill= np.insert(dly_diff, np.repeat(0,len(closep)-len(dly_diff)),np.zeros(len(closep)-len(dly_diff)) + np.nan)
    dly_return_fill= np.insert(dly_return_p, np.repeat(0,len(closep)-len(dly_return_p)),np.zeros(len(closep)-len(dly_return_p)) + np.nan)
    dr_mean = np.insert(dr_mean, np.repeat(0,len(closep)-len(dr_mean)),np.zeros(len(closep)-len(dr_mean)) + np.nan)
    up_band_fill= np.insert(up_band, np.repeat(0,len(closep)-len(up_band)),np.zeros(len(closep)-len(up_band)) + np.nan)
    low_band_fill= np.insert(low_band, np.repeat(0,len(closep)-len(low_band)),np.zeros(len(closep)-len(low_band)) + np.nan)
    
    Indicators = pd.DataFrame({
            'Date': history1year.Date,
            'Global_Mean': glMean,
            'SMA_5': mav5_fill,
            'SMA_20': mav20_fill,
            'smaSD_5': sd_5_fill,
            'smaSD_20': sd_20_fill,
            'Bollinger_Up': up_band_fill,
            'Bollinger_Low': low_band_fill,
            'Daily_Return(%)': dly_return_fill,
            'Daily_Return_Mean(%)': dr_mean,
            'Daily_Difference': dly_diff_fill,
            'Intraday_Range(C-O)': intraday_PriceRange,
            'Intraday_Difference(H-L)': intraday_PriceDiff,
            'MACD': macd[0],
            'MACD_single': macd[1],
            'MACD_histogram': macd[2]},
            columns=['Date','Global_Mean','SMA_5','SMA_20','smaSD_5','smaSD_20','Bollinger_Up',
                     'Bollinger_Low','Daily_Return(%)','Daily_Return_Mean(%)','Daily_Difference',
                     'Intraday_Range(C-O)','Intraday_Difference(H-L)','MACD','MACD_single','MACD_histogram']
            )
        
    studies_table = pd.merge(history1year,Indicators,how='inner',on='Date')    
    
    '''
    make the figure
    '''
      
    img = BytesIO()  # create the buffer
    
    rcParams['figure.figsize'] = 10, 7
    plt.style.use('seaborn-ticks')
    
    fig = plt.figure(facecolor='white')
    
    label1 = '5 Day SMA'
    label2 = '20 Day SMA'
    label3 = 'Upper Band'
    label4 = 'Lower Band'
    label5 = 'Global Mean'
    
    ax0 = plt.subplot2grid((18,4),(0,0),rowspan=4,colspan=4)
    candlestick_ohlc(ax0, his1Y_Ar, width=.6, colorup='g', colordown='r', alpha=0.75)
    ax0.plot(date[-start_20:],mav20[-start_20:], color='#9eff15', label=label2)
    ax0.plot(date[-start_20:],up_band[-start_20:], color='#0000ff', label=label3)
    ax0.plot(date[-start_20:],low_band[-start_20:], color='#ff1d81', label=label4)
    plt.ylabel('Bollinger Bands')
    plt.legend(loc=0,prop={'size':8},fancybox=True)
    
    # color settings
    ax0.set_facecolor("white")
    
    ax1 = plt.subplot2grid((18,4),(6,0),sharex=ax0,rowspan=8,colspan=4)
    ax1.spines['bottom'].set_color("#07000d")
    ax1.spines['top'].set_color("#07000d")
    ax1.spines['left'].set_color("#07000d")
    ax1.spines['right'].set_color("#07000d")
    ax1.set_frame_on(True)
    ax1.set_xticks(date)
    candlestick_ohlc(ax1, his1Y_Ar, width=.6, colorup='g', colordown='r', alpha=0.75)
    
    ax1.plot(date, glMean, color='k',linewidth=0.8,label=label5)
    
    ax1.errorbar(date[-start_5:],mav5[-start_5:], color='#cc4f1b', label=label1)
    ax1.fill_between(date[-start_5:], mav5[-start_5:]-sd_5, mav5[-start_5:]+sd_5, alpha=0.5, edgecolor='#cc4f1b', facecolor='#ff9848')
    
    ax1.errorbar(date[-start_20:],mav20[-start_20:], color='#66beff', label=label2)
    ax1.fill_between(date[-start_20:], mav20[-start_20:]-sd_20, mav20[-start_20:]+sd_20, alpha=0.5, edgecolor='#66beff', facecolor='#e5f4ff')
    
    ax1.grid(False)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(20))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('Stock Price')
    plt.legend(loc=4,prop={'size':8},fancybox=True)
    
    # color settings
    ax1.set_facecolor("white")
    ax1.spines['left'].set_color("#252525")
    ax1.spines['right'].set_color("#252525")
    ax1.spines['bottom'].set_color("#252525")
    ax1.spines['top'].set_color("#252525")
    ax1.xaxis.label.set_color("#252525")
    ax1.tick_params(axis='x', colors="#252525")
    ax1.yaxis.label.set_color("#252525")
    ax1.tick_params(axis='y', colors="#252525")
    
    ax1v = ax1.twiny()
    ax1v.hist(closep,orientation='horizontal', bins=20, alpha=0.2)
    ax1v.yaxis.labelpad = 0
    plt.xlabel('(Frequency of Volumn)',color='grey', fontsize=9)
    ax1v.xaxis.set_label_position('top')
        
    ax2 = plt.subplot2grid((18,4),(14,0),sharex=ax0,rowspan=2,colspan=4)
    ax2.plot(date,volume,color='#0094ff',linewidth=0.8)
    ax2.fill_between(date,0,volume,facecolor='#0094ff',alpha=.5)
    plt.ylabel('Volume')
    
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(20))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    
    # color settings
    ax2.set_facecolor("white")
    
    ax3 = plt.subplot2grid((18,4),(16,0),sharex=ax0,rowspan=2,colspan=4)
    ax3.plot(date,macd[0],color='#0094ff',linewidth=0.8)
    ax3.plot(date,macd[1],color='#0094ff',linewidth=0.8)
    ax3.fill_between(date,macd[0]-macd[1],0,facecolor='#29ff15',alpha=.5)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
    plt.ylabel('MACD')
    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(45)
    
    # color settings
    ax3.set_facecolor("white")
    
    plt.suptitle(str(stock).upper())
    plt.setp(ax0.get_xticklabels(),visible=False)
    plt.setp(ax1.get_xticklabels(),visible=False)
    plt.setp(ax2.get_xticklabels(),visible=False)
    plt.subplots_adjust(left=.10,bottom=.20,right=.95,top=0.92,wspace=.20,hspace=.08)
    
    plt.savefig(img, format='png',facecolor=fig.get_facecolor(), edgecolor='#07000d')
    # save figure to the buffer
    img.seek(0)  # rewind the buffer
    
    plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
    
    # round the number od the two tables to the 4 decimals
    tem = studies_table.copy()
    tem = tem.set_index('Date')
    tem.loc[:, tem.dtypes == 'float64']=tem.loc[:, tem.dtypes == 'float64'].apply(lambda x: round(x,4))
    tem = tem.reset_index()
    
    return render_template('view_t_c_studies.html', plot_url=plot_data, selectdic=sldic, tables=[tem.to_html(classes='history')], titles = ['na',''])

    
@app.route("/linear_regression",methods=['GET', 'POST'])
def weather_and_stock():
    try:
        '''
        get 1 year of historical data for the selected stock
        '''
        select = request.form.get('ticker_select')
        stock = str(select).lower()
        sldic = {select.upper()}
        url = 'http://www.nasdaq.com/symbol/'+stock+'/historical/'
        history1year = HisDataGain(url)
        
        '''
        get time-series data of NYC weather
        '''  
        location = 'KNYC'
        
        timedelta(days=1*365)
        oneYearAgo=(datetime.now() - timedelta(days=1*365)).strftime("%Y/%m/%d")
        
        currentDay = str(datetime.now().day)
        currentMonth = str(datetime.now().month)
        currentYear = str(datetime.now().year)
        
        url_3 = 'https://www.wunderground.com/history/airport/'+location+'/'+oneYearAgo+'/CustomHistory.html?dayend='+currentDay+'&monthend='+currentMonth+'&yearend='+currentYear+'&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo='
        
        weather = WeatherDataGain(url_3)
        
        '''
        combine the stock price data and the weather data
        '''    
        history_weather = weather.merge(history1year, how='left', left_on='Date', right_on='Date')
        history_weather.dropna(inplace=True)
        history_weather.columns.values[15] = 'Wind (mph) avg'
        history_weather.columns.values[16] = 'Wind (mph) low'
        cols = ['Temp. (°F) high', 'Temp. (°F) avg', 'Temp. (°F) low',
       'Dew Point (°F) high', 'Dew Point (°F) avg', 'Dew Point (°F) low',
       'Humidity (%) high', 'Humidity (%) avg', 'Humidity (%) low',
       'Sea Level Press. (in) high', 'Sea Level Press. (in) avg',
       'Sea Level Press. (in) low', 'Visibility (mi) high',
       'Visibility (mi) avg', 'Visibility (mi) low', 'Wind (mph) avg',
       'Wind (mph) low', 'Wind (mph) high', 'Precip. (in) sum' ]
       
        history_weather[cols] = history_weather[cols].replace({'-': np.nan,'T': np.nan}, regex=True)
        history_weather[cols] = history_weather[cols].astype(float) 
        
    except Exception as e:
        print('Error: ',e)
            
    '''
    make figure
    '''
    # use seaborn model to make the linear regression figure
    
    sns.set(color_codes=True)
    
    img = BytesIO()  # create the buffer
      
    r2 = Rsquared(history_weather)
    
    vars = ['Temp. (°F) avg','Dew Point (°F) avg', 'Humidity (%) avg', 'Sea Level Press. (in) avg',
            'Visibility (mi) avg', 'Wind (mph) avg','Precip. (in) sum']
    
    rcParams['figure.figsize'] = 12, 12
    plt.style.use('seaborn-ticks')
    fig, axs = plt.subplots(3,3)
    sns.regplot(x=vars[0], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[0])},dropna=True,ax=axs[0,0])
    # plot legend
    axs[0,0].legend(loc=1)
    sns.regplot(x=vars[1], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[1])},dropna=True, ax=axs[0,1])
    axs[0,1].legend(loc=1)
    sns.regplot(x=vars[2], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[2])},dropna=True, ax=axs[0,2])
    axs[0,2].legend(loc=1)
    sns.regplot(x=vars[3], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[3])},dropna=True, ax=axs[1,0])
    axs[1,0].legend(loc=1)
    sns.regplot(x=vars[4], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[4])},dropna=True, ax=axs[1,1])
    axs[1,1].legend(loc=1)
    sns.regplot(x=vars[5], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[5])},dropna=True, ax=axs[1,2])
    axs[1,2].legend(loc=1)
    sns.regplot(x=vars[6], y='Close / Last', data=history_weather,line_kws={'label':"$R^2={0:.4f}$".format(r2[6])},dropna=True, ax=axs[2,0])
    axs[2,0].legend(loc=1)
    
    plt.savefig(img, format='png')
    # save figure to the buffer
    img.seek(0)  # rewind the buffer
    
    plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
    
    return render_template('view_t_c_weather_lr.html', plot_url=plot_data, selectdic=sldic, tables=[history_weather.to_html(classes='history')], titles = ['na',''])

# Force Index  
def FI(df, n):
    F = pd.Series(df['Close / Last'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    return F

def DataForModel(history1year):
    stockfile = history1year.values.T.tolist()
    date = mdates.date2num(stockfile[0])   
    openp = np.loadtxt(stockfile[1], delimiter=',', unpack=True)
    highp = np.loadtxt(stockfile[2], delimiter=',',  unpack=True)
    lowp = np.loadtxt(stockfile[3], delimiter=',', unpack=True)
    closep = np.loadtxt(stockfile[4], delimiter=',', unpack=True)
    volume = np.loadtxt(stockfile[5], delimiter=',', unpack=True)
    
    '''
    Generate features
    '''
    ForceIndex = FI(history1year,14)
    
    # William %R
    WILLR = talib.WILLR(highp, lowp, closep, timeperiod=14)
    
    # Relative Strength Index(RSI)
    RSI = talib.RSI(closep, timeperiod=14)
    
    # Rate Of Change(ROC), this is daily return too
    ROC = talib.ROC(closep, timeperiod=14)
    
    # Momentum (MOM)
    MOM = talib.MOM(closep, timeperiod=14)
    
    # Average True Range(ATR)
    ATR = talib.ATR(highp,lowp,closep)  
    
    #Parabolic SAR 
    SAR = talib.SAR(highp, lowp, acceleration=0.02, maximum=0.2)
    
    #Price range for the day (bars)
    intraday_PriceRange = closep- openp
    
    # combine the statistics into one dataframe
    mlFeatures = pd.DataFrame({ 
            'Date': history1year.Date,
            'Close / Last': closep,
            'Intraday_Range(C-O)': intraday_PriceRange,
            'FI':ForceIndex,
            'William %R': WILLR,
            'RSI': RSI,
            'ROC': ROC,
            'MOM': MOM,
            'ATR': ATR,
            'MACD': MACD(history1year, 12, 26)[0],
            'SAR': SAR},
            columns=['Date','Close / Last','Intraday_Range(C-O)','FI','William %R','RSI','ROC','MOM','ATR','MACD','SAR']
            )
        
    mlFeatures['Up/Down'] = np.where(mlFeatures['Intraday_Range(C-O)']>0, 'up', 'down')
    
    mlFeatures = mlFeatures.set_index('Date')
    
    return mlFeatures

@app.route("/prediction", methods=['GET', 'POST'])
def predict():
    
    '''
    the data containing historical price and indicators
    ''' 
    mlFeatures = pd.DataFrame()
    plot_data = str()
    select = request.form.get('ticker_select')
    stock = str(select).lower()
    sldic = {select.upper()}
    url = 'http://www.nasdaq.com/symbol/'+stock+'/historical/'
    
    history1year = HisDataGain(url)

    mlFeatures = DataForModel(history1year)
    
    '''
    SVM model:classification
    '''
    mlFeatures_C = DataForModel(history1year)

    mlFeatures_C.fillna(-9999,inplace=True)
    mlFeatures_C.dropna(inplace=True)
    
    X = np.array(mlFeatures_C.drop(['Intraday_Range(C-O)','Up/Down'],1))
    X = preprocessing.scale(X)
    y = mlFeatures_C['Up/Down']
    
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    
    svc_lin = svm.SVC(kernel='linear', C=1, gamma=1) 
    svc_lin.fit(X_train,y_train)
    accuracy_svclin = svc_lin.score(X_test,y_test)
    print(accuracy_svclin)
    svc_lin.predict(X_test)
    
    forecast_period = int(math.ceil(0.1*len(mlFeatures_C)))
    
    X = X[:-forecast_period]
    X_future = X[-forecast_period:]
    forecast_class = svc_lin.predict(X_future)
    
    mlFeatures_C['Predicted Up/Down'] = np.nan
        
    mlFeatures_C.sort_index(axis=0, ascending=True,inplace=True)
               
    last_date = mlFeatures_C.iloc[-1].name
    last_timestamp = last_date.timestamp()
    one_day = 86400
    next_timestamp = last_timestamp + one_day
    
    for i in forecast_class:
        next_date = datetime.fromtimestamp(next_timestamp)
        next_timestamp += one_day
        mlFeatures_C.loc[next_date] = [np.nan for _ in range(len(mlFeatures_C.columns)-1)]+[i]
    
    '''
    SVM model:regression
    '''
    mlFeatures = DataForModel(history1year)
    
    mlFeatures.fillna(-9999,inplace=True)
    forecast_label = 'Close / Last'
    forecast_period = int(math.ceil(0.1*len(mlFeatures)))
    
    mlFeatures['label'] = mlFeatures[forecast_label].shift(-forecast_period)
    mlFeatures.dropna(inplace=True)
    
    X = np.array(mlFeatures.drop(['Close / Last','Up/Down'],1))
    X = preprocessing.scale(X)
    y = mlFeatures['Close / Last']
    
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    
    svr_lin = svm.SVR(kernel='linear',C=1e3)
    svr_pol = svm.SVR(kernel='poly',C=1e3,degree=2)
    svr_rbf = svm.SVR(kernel='rbf',gamma=0.1)
    
    with open('svm_linear_regression.pickle','wb') as f:
        pickle.dump(svr_lin,f)
    pickle_model = open('svm_linear_regression.pickle','rb')
    svr_lin = pickle.load(pickle_model)
    
    with open('svm_poly_regression.pickle','wb') as f:
        pickle.dump(svr_pol,f)
    pickle_model = open('svm_poly_regression.pickle','rb')
    svr_pol = pickle.load(pickle_model)
    
    with open('svm_rbf_regression.pickle','wb') as f:
        pickle.dump(svr_rbf,f)
    pickle_model = open('svm_rbf_regression.pickle','rb')
    svr_rbf = pickle.load(pickle_model)
    
    svr_lin.fit(X_train,y_train)
    svr_pol.fit(X_train,y_train)
    svr_rbf.fit(X_train,y_train)
    
    accuracy_svrlin = svr_lin.score(X_test,y_test)
    accuracy_svrpol = svr_pol.score(X_test,y_test)
    accuracy_svrrbf = svr_rbf.score(X_test,y_test)
    print(accuracy_svrlin,accuracy_svrpol,accuracy_svrrbf)
    
    forecast_period = int(math.ceil(0.1*len(mlFeatures)))
    
    X = X[:-forecast_period]
    X_future = X[-forecast_period:]
    forecast_set = svr_lin.predict(X_future)
    
    mlFeatures['Predicted price'] = np.nan
    
    mlFeatures.sort_index(axis=0, ascending=True,inplace=True)
               
    last_date = mlFeatures.iloc[-1].name
    last_timestamp = last_date.timestamp()
    one_day = 86400
    next_timestamp = last_timestamp + one_day
    
    for i in forecast_set:
        next_date = datetime.fromtimestamp(next_timestamp)
        next_timestamp += one_day
        mlFeatures.loc[next_date] = [np.nan for _ in range(len(mlFeatures.columns)-1)]+[i]
    
    '''
    combine classcification and regression table
    '''
    cols = ['label','Predicted price']
    mlFeatures_C[cols] = mlFeatures[cols] 
    
    '''
    make figure
    '''
    mlFeatures.drop(['Up/Down'],1,inplace=True)
    mlFeatures = mlFeatures.apply(lambda x: round(x,4))
    mlFeatures = mlFeatures.reset_index()
    
    img = BytesIO()  # create the buffer
    
    matplotlib.rcParams.update({'font.size':12})
    
    label1 = 'Real Price'
    label2 = 'Predict Price'
                
    fig = plt.figure
    fig, ax = plt.subplots(figsize=(10,6),facecolor='w')
    fig.subplots_adjust(bottom=0.2)
    ax.plot(mlFeatures['Date'],mlFeatures['Close / Last'],color='#0000ff',label=label1)
    ax.plot(mlFeatures['Date'],mlFeatures['Predicted price'], color='#ff1493', label=label2)
    #dates on the x-axis
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(),rotation=60)
    
    plt.legend(loc=0,prop={'size':9},fancybox=True)
    plt.ylabel('Stock Price')
    plt.grid(linestyle='dotted')
    
    plt.savefig(img, format='png')
    # save figure to the buffer
    img.seek(0)  # rewind the buffer
    
    plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
    
    return render_template('view_t_c_ml.html', plot_url=plot_data, selectdic=sldic, tables=[mlFeatures_C.to_html(classes='history')], titles = ['na',''])
    

@app.route("/100days_history", methods=['GET', 'POST'])
def days100():
    errors = []
    select = request.form.get('ticker_select')
    select = str(select).lower()
    # get the data
    end = datetime.now()
    N = 145
    date_N_days_ago = datetime.now() - timedelta(days=N)
    his100 = pd.DataFrame()
    sldic = {select.upper()}
    plot_data = str()
    col_Date = pd.Series()
    stat = pd.DataFrame()
    if request.method == "POST":
        # get url that the user has entered
        try:
            his100 = pdr.DataReader(select, 'yahoo', date_N_days_ago, end)
             # convert the date index to column
            his100.reset_index(level=0, inplace=True)
            col_Date = his100.Date
            # change the date to the number format to generate the plot
            his100.Date = mdates.date2num(his100.Date.dt.to_pydatetime())
            
            # prepare array for the condlestick chart
            his100_Ar = [tuple(x) for x in his100[['Date', 'Open', 'Close', 'High', 'Low']].to_records(index=False)]
            
            # draw the figure
            img = BytesIO()  # create the buffer
            
            rcParams['figure.figsize'] = 5, 4
            plt.style.use('seaborn-ticks')
            
            fig = plt.figure(facecolor='white')
            ax1 = plt.subplot2grid((5,4),(0,0),rowspan=4,colspan=4)
            candlestick_ochl(ax1, his100_Ar, width=.6, colorup='#0000ff', colordown='#ff0000', alpha=0.75)
            ax1.grid(True,linestyle='dotted')
            ax1.xaxis.set_major_locator(mticker.MaxNLocator(20))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.ylabel('Stock Price')
            
            # color settings
            ax1.set_facecolor("white")
            
            ax2 = plt.subplot2grid((5,4),(4,0),sharex=ax1,rowspan=1,colspan=4)
            ax2.plot(his100[['Date']], his100[['Volume']] ,color='#0094ff',linewidth=0.8)
            ax2.fill_between(his100['Date'],0, his100['Volume'],facecolor='#0094ff',alpha=.4)
            plt.ylabel('Volume')
                
            #ax2.xaxis.set_major_locator(mticker.MaxNLocator(20))
            #ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            #ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
            #for label in ax2.xaxis.get_ticklabels():
            #    label.set_rotation(45)
            ax2.xaxis_date()
            ax2.autoscale_view()
            plt.setp(plt.gca().get_xticklabels(),rotation=45)
                
            # color settings
            ax2.set_facecolor("white")
            
            # hide x axis tick label for ax1
            plt.setp(ax1.get_xticklabels(),visible=False)
            
            plt.subplots_adjust(left=.10,bottom=.20,right=.95,top=0.92,wspace=.20,hspace=.005)

            plt.savefig(img, format='png',facecolor=fig.get_facecolor(), edgecolor='#07000d')

            # save figure to the buffer
            img.seek(0)  # rewind the buffer
            
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
            
        except:
            errors.append(
                "Unable to get market price. Please make sure the ticker symbol is valid and try again."
            )
    # change the date to the datetime format
    h100 = his100
    h100.Date = col_Date
    
    # basic analysis
    stat = h100.loc[:, h100.columns != 'Date'].describe()
    
    # round the number od the two tables to the 4 decimals
    tem=h100
    tem=tem.set_index('Date')
    tem=tem.apply(lambda x: round(x,4))
    tem=tem.reset_index()
    stat=stat.apply(lambda x: round(x,4))

    return render_template('view_t_c_for3.html', plot_url=plot_data, selectdic=sldic, tables=[tem.to_html(classes='history'),stat.to_html(classes='anl')], titles = ['na','',''])

@app.route('/blotter/')
def select_blotter():
    return redirect(url_for("show_blotter"))

@app.route('/blotter/show')
def show_blotter():
    # get blotter and pl data from database(here I use AWS RDS MySQL engine)
    # get blotter and pl data from database(here I use AWS RDS MySQL engine)
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306')
    engine.execute("CREATE DATABASE IF NOT EXISTS trade_data") #create db
    engine.execute("USE trade_data") # select new db
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306/trade_data')
    conn = engine.connect()
    try:
        blotter = pd.read_sql('blotter', conn)
    except:
        blotter = pd.read_sql('blotter', conn)

    conn.close()
    
    blotter = blotter.set_index('index')
    blotter.index.name = None
    
    return render_template('viewblotter.html',tables=[blotter.to_html(classes='blt')],titles = ['','']) 

@app.route('/pl/')
def select_pl():
   return redirect(url_for("show_pl"))

@app.route('/pl/show')
def show_pl():
    # get blotter and pl data from database(here I use AWS RDS MySQL engine)
    # get blotter and pl data from database(here I use AWS RDS MySQL engine)
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306')
    engine.execute("CREATE DATABASE IF NOT EXISTS trade_data") #create db
    engine.execute("USE trade_data") # select new db
    engine=create_engine('mysql://yun:cuny1234@cunysps.cvbiwzsoozos.us-east-1.rds.amazonaws.com:3306/trade_data')
    conn = engine.connect()
    try:
        pl = pd.read_sql('pl', conn)
    except:
        pl = pd.read_sql('pl', conn)
        
    conn.close()
       
    pl = pl.set_index('index')
    pl.index.name = None
    
    return render_template('viewpl.html',tables=[pl.to_html(classes='pl')],titles = ['','']) 
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',use_reloader=False)
    

