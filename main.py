#import bs4 as bs
#import datetime as dt
#import matplotlib.pyplot as plt
#from matplotlib import style
#from collections import Counter
#import numpy as np
#import os
#import pandas as pd
#import pandas_datareader.data as web
#import pickle
#import requests
#
#from sklearn import svm, neighbors
#from sklearn.ensemble import VotingClassifier, RandomForestClassifier
#from sklearn.model_selection import train_test_split
#
#style.use('ggplot')

import datetime as dt
import multiprocessing
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


import FMP_data as fmp
import yahoo_yfinance as yahoo
from settings import settings
import myPortfolio
import myStocks

tickers = fmp.get_tickers1()
 
def get_data(tickers, yahooData=True, start_date='1999-01-01', end_date=dt.datetime.now().strftime('%Y-%m-%d'), update=False, update_info=False):
    if update_info:
            fmp.get_info(tickers)
    
    if yahooData:
        tickers += settings.etf_tickers
        for ticker in tickers:
            if not update:
                yahoo.new_data_yahoo(ticker, start_date, end_date)
            if update:
                yahoo.update_data_yahoo(ticker, end_date)
    else:
        for ticker in tickers:
            if not update:
                fmp.new_data(ticker, start_date, end_date)
            if update:
                fmp.update_data(ticker, end_date)          
    
#def compile_data(tickers, yahooData=True):
#
#    main_df = pd.DataFrame()
#    
#    if yahooData:
#        tickers_all = tickers + settings.etf_tickers
#        rw_path = settings.Yahoo_Data_Path
#        date_column = settings.yahoo_date
#        close_column = settings.yahoo_close
#        drop_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#    else:
#        tickers_all = tickers
#        rw_path = settings.FMP_data
#        date_column = settings.FMP_date
#        close_column = settings.FMP_close
#        drop_columns = ['index','open','high','low','volume','unadjustedVolume','change','changePercent','vwap','label','changeOverTime']
#        
#    for ticker in tickers_all:
#        if os.path.exists(rw_path + '/{}.csv'.format(ticker)):   
#            df = pd.read_csv(rw_path + '/{}.csv'.format(ticker))
#            if not df.empty:
#                df.set_index(date_column, inplace=True)
#                
#                df.rename(columns={close_column: ticker}, inplace=True)
#                df.drop(drop_columns, 1, inplace=True)
#                
#                if main_df.empty:
#                    main_df = df
#                else:
#                    main_df = main_df.join(df, how='outer')
#                    print('Adding {}'.format(ticker))
#                        
#    main_df.to_csv(rw_path + '/joined_closes.csv')
    
def compile_data(tickers, yahooData=True):
    df_current = pd.DataFrame()

    if yahooData:
        tickers_all = tickers + settings.etf_tickers
        rw_path = settings.Yahoo_Data_Path
        date_column = settings.yahoo_date
        close_column = settings.yahoo_close
        drop_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        tickers_all = tickers
        rw_path = settings.FMP_data
        date_column = settings.FMP_date
        close_column = settings.FMP_close
        drop_columns = ['index','open','high','low','volume','unadjustedVolume','change','changePercent','vwap','label','changeOverTime']
    
    if os.path.exists(rw_path + '/joined_closes.csv'):    
        df_current = pd.read_csv(rw_path + '/joined_closes.csv')
        df_current.set_index(date_column, inplace=True)                
        current_tickers = df_current.columns.values.tolist()
        for current in current_tickers:
            if current in tickers_all:
                tickers_all.remove(current)

    for ticker in tickers_all:
        if os.path.exists(rw_path + '/{}.csv'.format(ticker)):
            df = pd.read_csv(rw_path + '/{}.csv'.format(ticker))
            if not df.empty:
                
                if df_current.empty:
                    df_current = df
                    df_current.set_index(date_column, inplace=True)                
                    df_current.rename(columns={close_column: ticker}, inplace=True)
                    df_current.drop(drop_columns, 1, inplace=True)
                else:
                    df.set_index(date_column, inplace=True)                
                    df_current[ticker] = df[close_column]
                print('Adding {}'.format(ticker))                    
    
    df_current.to_csv(rw_path + '/joined_closes.csv')

def visualize_data(yahooData=True):
    if yahooData:
        path = settings.Yahoo_Data_Path
        date_column = 'Date'
    else:
        path = settings.FMP_data
        date_column = 'date'
    
    if not os.path.exists(path + '/joined_closes.csv') and not os.path.exists(path + '/joined_corr.csv'):    
        df = pd.read_csv(path + '/joined_closes.csv')
    
        #df_corr = df.corr()
        df.set_index(date_column, inplace=True) 
        df_corr = df.pct_change().corr() 
        df_corr.to_csv(path + '/joined_corr.csv')
        data1 = df_corr.values
    else:
        df_corr = pd.read_csv(path + '/joined_corr.csv')
        data1 = df_corr.values
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
    
def combinations(tickers, r=2):
    return list(itertools.combinations(tickers, r)) 
        

def main():
    '''
    set up parameters required by the task
    '''
    n_processors = 8
    

    '''
    pass the task function, followed by the parameters to processors
    '''    
#    with multiprocessing.Pool(processes=n_processors) as pool:
#        results = pool.map(compile_data, tickers[600:650])
#      
if __name__ == "__main__":        

    multiprocessing.freeze_support()   # required to use multiprocessing
    main()
    
#a = myPortfolio.Portfolio(stocks=['EMIM.AS', 'GE', 'AAPL', 'MSFT'], weights=[0.20,0.20, 0.5, 0.1])

#a = myPortfolio.Portfolio(stocks=['EMIM.AS', 'IBZL.AS', 'CSPX.AS', 'IJPA.AS', 'CSX5.AS', 'CEMU.AS'], weights=[0.35, 0.45, 0.05, 0.05, 0.10], start_date='2019-01-01')

a = myPortfolio.Portfolio(stocks=['IWDA.AS', 'EMIM.AS', 'SXRG.DE'], weights=[0.88, 0.12, 0.0], start_date='2012-01-01', end_date='2018-12-31')
b = myPortfolio.Portfolio(stocks=['IWDA.AS', 'SXRG.DE'], weights=[0.65, 0.35], start_date='2012-01-01', end_date='2018-12-31')
c = myPortfolio.Portfolio(stocks=['IWDA.AS', 'SXRG.DE', 'LQDA.AS'], weights=[0.65, 0.35, 0.0], start_date='2012-01-01', end_date='2018-12-31')
d = myPortfolio.Portfolio(stocks=['IQQ0.DE', 'SXRG.DE'], weights=[0.88, 0.12], start_date='2012-01-01', end_date='2018-12-31')
#a = myPortfolio.Portfolio(stocks=['IWDA.AS', 'EMIM.AS'], weights=[0.88, 0.12], start_date='2017-01-01', end_date='2019-11-26')
#a = myPortfolio.Portfolio(stocks=['BAC', 'IBM', 'AAL'], weights=[0.88, 0.12, 0.0], start_date='2000-01-01', end_date='2018-12-31')
#a.run_cppi(a.gbm(numberofPortfolios=10, graph=False))

z = myPortfolio.Portfolio(stocks=['IWDA.AS', 'EMIM.AS'], weights=[0.88, 0.12], start_date='2010-01-01', end_date='2020-01-01')
A = myPortfolio.Portfolio(stocks=['IWDA.AS'], weights=[1], start_date='2010-01-01', end_date='2020-01-01')
B = myPortfolio.Portfolio(stocks=['SPPW.DE'], weights=[1], start_date='2010-01-01', end_date='2020-01-01')
C = myPortfolio.Portfolio(stocks=['^GSPC'], weights=[1], start_date='1999-01-01', end_date='2020-01-01')


def investment(*portfolios, period, period_amount, years, annualized_growth=None):
    portfolio_compare = []
    fees_portfolio = []
    for portfolio in portfolios:
        #portfolio = portfolio.otimizedPortfolio(numberOfPortfolios=2, index_show=False)
        if annualized_growth is None:
            annualized_growth = portfolio.annualized_return()
        
        total_sold = []
        total_fee = []
        
        index = 0
        for ticker in portfolio.tickers:
            total = 0
            fee_amount = 0
            fee = settings.etf_tickers[ticker]['Fee']
            degiro_free = settings.etf_tickers[ticker]['Degiro Free']
            weighted_amount = portfolio.weights[index] * period_amount
            for i in range (years):
                if not degiro_free:
                    fee_transaction = settings.etf_transaction_fee + (weighted_amount * (settings.etf_fee/100))
                    fee_amount += (period * fee_transaction) + settings.exchange_year
                    real_amount_period = weighted_amount - fee_transaction
                else:
                    real_amount_period = weighted_amount

                annualized_amount = period * real_amount_period
                total = (total + annualized_amount) * (1+annualized_growth)*(1-(fee/100))
                                        
            if not degiro_free:
                fee_transaction = settings.etf_transaction_fee + (total * (settings.etf_fee/100))
                fee_amount += fee_transaction
                total_sold.append(total - fee_transaction)
            else:
                total_sold.append(total)            
            total_fee.append(fee_amount)
            index += 1
        
        portfolio_compare.append(sum(total_sold))
        fees_portfolio.append(sum(total_fee))

        #print ('Portfolio', index, ' - ', 'Amount When Sold: ', '{:.2f}'.format(sum(total_sold)), '€')
        #print ('Portfolio', index, ' - ', 'Amount After Taxes: ', '{:.2f}'.format(sum(total_sold) - sum(total_sold)*settings.portugal_tax), '€')
    
    print ('Amount Invested:', period*period_amount*years, '€')
    for amount,fees in zip(portfolio_compare, fees_portfolio):    
        print ('Amount When Sold:', '{:.2f}'.format(amount), '€', 'Total Degiro Fees:', '{:.2f}'.format(fees), '€', 'Amount after Taxes:', '{:.2f}'.format(amount*(1-settings.portugal_tax)), '€')

#investment(A,B,z, period=12, period_amount=100, years=20, annualized_growth=C.annualized_return())
us_portfolio = myPortfolio.Portfolio(stocks=['XOM', 'TAST', 'BAC'], start_date='2010-01-01', end_date='2020-01-01')

#get_data(tickers, end_date='2020-01-01', update_info=True, update=True)