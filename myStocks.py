#Module Stocks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

from settings import settings

class Stock:
    def __init__(self, ticker, yahoo=True):
        self.symbol = ticker
        if yahoo:
            self.data = pd.read_csv(settings.Yahoo_Data_Path + '/{}.csv'.format(ticker))
            self.close = 'Adj Close'
            self.volume = 'Volume'
            self.date = 'Date'
        
        if not yahoo:
            self.data = pd.read_csv(settings.FMP_data + '/{}.csv'.format(ticker))
            self.close = 'close'
            self.volume = 'volume'
            self.date = 'date'
            
        self.info_data = pd.read_csv(settings.info_file_path)
        
        if not self.info_data.loc[(self.info_data["Unnamed: 0"] == 'companyName') & (self.info_data["symbol"] == ticker), "profile"].empty:
            self.name = self.info_data.loc[(self.info_data["Unnamed: 0"] == 'companyName') & (self.info_data["symbol"] == ticker), "profile"].values[0]
        else:
            self.name = self.symbol
        
    def lastClose(self):
        return self.data[self.close].iloc[-1]
    
    
    def graph(self, days=10, **kwargs):
        df = self.data.copy()
            
        df[self.date] = pd.to_datetime(df[self.date])
        df.reset_index(inplace=True)
        df.set_index(self.date, inplace=True)  
        
        ma = kwargs.get('ma')
        if ma:
            for x in ma:
                df[str(x)+'ma'] = df[self.close].rolling(window=x, min_periods=0).mean()
        
        df_ohlc = df[self.close].resample(str(days)+'D').ohlc()
        df_volume = df[self.volume].resample(str(days)+'D').sum()
        df_ohlc.reset_index(inplace=True)
        df_ohlc[self.date] = df_ohlc[self.date].map(mdates.date2num)
        
        ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()
        
        candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
        
        if ma:
            for x in ma:
                ax1.plot(df.index, df[str(x)+'ma'], label=str(x)+'ma')
     
        ax1.legend(loc="upper left") 
        ax1.set_title(self.name)
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
        plt.show()
        
    def dataframe(self):
        df = self.data.copy()
        df[self.date] = pd.to_datetime(df[self.date])
        df.reset_index(inplace=True)
        df.set_index(self.date, inplace=True)
        return df
    
    def returns(self, resample_time=False):
        df = self.data.copy()
        df[self.date] = pd.to_datetime(df[self.date])
        df.reset_index(inplace=True)
        df.set_index(self.date, inplace=True)  
        
        if resample_time:
            returns = df[self.close].resample(resample_time).ffill().pct_change()
        else:
            returns = df[self.close].pct_change()
        return returns
    
    def returns_histogram(self, resample_time=False):
        
        if not resample_time:
            resample_time_string = '' 
        else:
            resample_time_string = resample_time
            
        returns = self.returns(resample_time)
        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        returns.plot.hist(bins = 60)
        ax1.set_xlabel(resample_time_string + " returns %")
        ax1.set_ylabel("Percent")
        ax1.set_title(self.name + ' ' + resample_time_string + " returns data")
        ax1.text(-0.15,200,"Extreme Low\nreturns")
        ax1.text(0.15,200,"Extreme High\nreturns")
        plt.show()
        
    def cumulative_returns(self, resample_time=False):      
        cum_returns = (self.returns(resample_time) + 1).cumprod()
        return cum_returns
       
    def cumulative_returns_graph(self, resample_time=False):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        plt.plot(self.cumulative_returns())
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Growth of $1 investment")
        ax1.set_title(self.name + ' ' + resample_time +" cumulative returns data")
        plt.show()
        