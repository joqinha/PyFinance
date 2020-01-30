#FMP DATA MODULE
import datetime as dt
import requests
import pandas as pd
import os

from itertools import product
import multiprocessing

from settings import settings

def get_tickers():
    tickers=[]
    
    url_list = 'https://financialmodelingprep.com/api/v3/company/stock/list'
    resp = requests.get(url=url_list)
    data = resp.json()
    for i in range(len(data['symbolsList'])):
        tickers.append(data['symbolsList'][i].get('symbol'))
    tickers.remove('PRN')                                           ##BECAUSE WINDOWS
    return tickers

def get_tickers1():
    df = pd.read_csv(settings.info_file_path)
    df = df.loc[(df["Unnamed: 0"] == 'companyName'), "symbol"]
    return df.values.tolist()
    

def get_info(tickers, file_path=settings.info_file_path):
    info_file_path = file_path
    
    main_df = pd.DataFrame()
            
    for ticker in tickers:
        print('Getting info data for {}'.format(ticker))
        url_profile = 'https://financialmodelingprep.com/api/v3/company/profile/'
        url = url_profile + ticker
        resp = requests.get(url=url)
        data = resp.json()
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.drop(['beta', 'changes', 'changesPercentage', 'image', 'lastDiv', 'mktCap', 'price', 'range', 'volAvg'], inplace=True)            
            main_df = pd.concat([main_df, df]) 
                
    main_df.to_csv(settings.info_file_path) 

def new_data(ticker, start_date=settings.date0, end_date=dt.datetime.now().strftime('%Y-%m-%d')):
    if not os.path.exists(settings.FMP_data):
        os.makedirs(settings.FMP_data)
        
    if not os.path.exists(settings.FMP_data + '/{}.csv'.format(ticker)):  
        url_historical_price = 'https://financialmodelingprep.com/api/v3/historical-price-full/'
        url = url_historical_price + ticker + '?from=' + start_date + '&to=' + end_date
        resp = requests.get(url=url)
        data = resp.json()
        
        df = pd.DataFrame(data)
        if not df.empty:    
            df = pd.DataFrame(data['historical'])
            df.reset_index(inplace=True)
            df.set_index('date', inplace=True)
            df.drop(['label', 'index'], axis=1, inplace=True)        
            df.to_csv(settings.FMP_data + '/{}.csv'.format(ticker))

def update_data(ticker, end_date=dt.datetime.now().strftime('%Y-%m-%d')):      
    end = end_date
    
    if os.path.exists(settings.FMP_data + '/{}.csv'.format(ticker)):   
        df = pd.read_csv(settings.FMP_data + '/{}.csv'.format(ticker))
        
        if not df.empty:
            df.set_index('date', inplace=True)
        
            start = df.index[-1]
            start_obj = dt.datetime.strptime(start, '%Y-%m-%d')
            start_obj += dt.timedelta(days=1)
            start = start_obj.strftime('%Y-%m-%d')
        
            if end != start:
                
                print('Updating {}'.format(ticker))
                
                url_historical_price = 'https://financialmodelingprep.com/api/v3/historical-price-full/'
                url = url_historical_price + ticker + '?from=' + start + '&to=' + end
                resp = requests.get(url=url)
                data = resp.json()
                
                df_test = pd.DataFrame(data)
                if not df_test.empty: 
                    df_update = pd.DataFrame(data['historical'])
                    df_update.reset_index(inplace=True)
                    df_update.set_index('date', inplace=True)
                    df_update = df_update.drop(['label', 'index'], axis=1)
                    df = pd.concat([df, df_update])
                    
                    df.to_csv(settings.FMP_data + '/{}.csv'.format(ticker))
                    
        else:
            new_data(ticker)