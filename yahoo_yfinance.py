#YAHOO DATA WITH YFINANCE
import yfinance as yf
import pandas as pd
import datetime as dt
import os

from settings import settings

def new_data_yahoo(ticker, start_date=settings.date0, end_date=dt.datetime.now().strftime('%Y-%m-%d')):
    
    if not os.path.exists(settings.Yahoo_Data_Path):
        os.makedirs(settings.Yahoo_Data_Path)
    
    if not os.path.exists(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker)):
        data = yf.download(ticker, start=start_date, end=end_date, group_by="ticker")
        if not data.empty and ticker != 'PRN':
            data.to_csv(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker))
    
def update_data_yahoo(ticker, end_date=dt.datetime.now().strftime('%Y-%m-%d')):
    end = end_date
    if os.path.exists(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker)):
        df = pd.read_csv(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        
        start = df.index[-1]
        start_obj = dt.datetime.strptime(start, '%Y-%m-%d')
        start_obj += dt.timedelta(days=1)
        start = start_obj.strftime('%Y-%m-%d')
        
        if end != start:
            df_update = yf.download(ticker, start=start, end=end, group_by="ticker")            
            df = pd.concat([df, df_update])        
            df.to_csv(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker))
            
    else:
        new_data_yahoo(ticker)
        
#for ticker in settings.etf_tickers:
#    new_data_yahoo(ticker, start_date='1999-01-01', end_date='2020-01-01')