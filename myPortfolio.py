import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import itertools
import ipywidgets as widgets
from IPython.display import display

import myStocks
from settings import settings

class Portfolio:
        
    def __init__(self, **kwargs):
        self.portfolio = []
        self.tickers = []
        
        self.start_date = kwargs.get('start_date', '2019-01-01')
        self.end_date = kwargs.get('end_date', '2018-12-31')
        
        yahoo = True
        if kwargs.get('yahoo') == True:
            yahoo = False
            
        if yahoo:
            drop_columns = settings.yahoo_drop_columns
        else:
            drop_columns = settings.fmp_drop_columns
        
        for stock in kwargs.get('stocks'):
            ticker = myStocks.Stock(stock, yahoo=yahoo)            
            self.portfolio.append(ticker)
            self.tickers.append(stock)
            
        self.portfolio_close_data = pd.DataFrame()
        for stock in self.portfolio:
            if self.portfolio_close_data.empty:
                self.portfolio_close_data = stock.dataframe()
                self.portfolio_close_data = self.portfolio_close_data[self.start_date:self.end_date]
                self.portfolio_close_data.rename(columns={stock.close: stock.symbol}, inplace=True)
                self.portfolio_close_data.drop(drop_columns, 1, inplace=True)
            else:
                self.portfolio_close_data[stock.symbol] = stock.dataframe()[stock.close]
                
        self.weights=[]
        if 'weights' in kwargs:
            for weight in kwargs.get('weights'):
                self.weights.append(weight)
        else:
            self.weights = np.repeat(1/len(self.tickers), len(self.tickers)).tolist()
        
    def ticker_graph(self):
        fig = plt.subplot()
        for ticker in self.portfolio:
            fig.plot(ticker.dataframe()[self.start_date:self.end_date].index, ticker.dataframe()[self.start_date:self.end_date][ticker.close], label=ticker.name)
        fig.xaxis_date()
        fig.autoscale_view()
        fig.legend(loc="upper left") 
        fig.set_title('Portfolio')
        plt.show()
    
    def individual_returns(self, resample_time=False):
        returns_portfolio = pd.DataFrame()      
        for stock in self.portfolio:
            df = pd.DataFrame(stock.returns(resample_time)[self.start_date:self.end_date])
            df = df.rename({stock.close: stock.symbol}, axis=1)
            returns_portfolio[stock.symbol] = df[stock.symbol]
        return returns_portfolio
    
    def returns(self, weights=None, resample_time=False):
        if weights is None:
            weights = self.weights
        weights = np.array(weights)
        weighted_returns = (weights.T * self.individual_returns(resample_time=resample_time))
        #weighted_returns = (weights.T * (self.individual_returns(resample_time=resample_time).mean() * 252))
        #teste = np.dot(weights.T, self.individual_returns(resample_time=resample_time, start_date=start_date))
        return weighted_returns
    
    def annualized_return(self, weights=None, resample_time=False, frequency=252):
        if weights is None:
            weights = self.weights
        return ((self.returns(weights, resample_time=resample_time).mean().sum()) * frequency)
    
    def volatility(self, weights=None, resample_time=False, frequency=252):
        if weights is None:
            weights = self.weights
        weights = np.array(weights)
        returns = self.returns(resample_time=resample_time, weights=weights)
        cov_returns = returns.cov()
        cov_returns *= frequency
        return np.sqrt(np.dot(weights.T, np.dot(cov_returns, weights)))
                 
    def sharpe_ratio_annualized(self, weights=None, RiskFree_Rate=0.02):
        if weights is None:
            weights = self.weights
        weights = np.array(weights)
        return ((self.annualized_return(weights) - RiskFree_Rate) / self.volatility(weights))

    def cumulative_returns(self, weights=None, resample_time=False, budget=1):
        if weights is None:
            weights = self.weights
        weights = np.array(weights)
        weighted_returns = self.returns(resample_time=resample_time, weights=weights)
        weighted_returns_sum = weighted_returns.sum(axis=1)
        cum_returns = (weighted_returns_sum + 1).cumprod()
        return budget * cum_returns
    
    def max_drawdown(self, resample_time=False):
        cum_returns = self.cumulative_returns(resample_time=resample_time)
        previous_peak = cum_returns.cummax()
        drawdown = (cum_returns - previous_peak) / previous_peak
        return drawdown.min()
    
    def index_returns(self, index_ticker='^GSPC', resample_time=False, budget=1):
        index = myStocks.Stock(index_ticker)
        returns_index = pd.DataFrame()      
        df = pd.DataFrame(index.returns(resample_time))
        df = df.rename({index.close: index_ticker}, axis=1)
        returns_index[index.symbol] = df[self.start_date:][index.symbol]
        index_returns_sum = returns_index.sum(axis=1)
        cum_returns = (index_returns_sum + 1).cumprod()
        return budget * cum_returns, returns_index
    
    def index_max_drawdown(self, index_ticker='^GSPC', resample_time=False):
        cum_returns, x = self.index_returns(index_ticker=index_ticker, resample_time=resample_time)
        previous_peak = cum_returns.cummax()
        drawdown = (cum_returns - previous_peak) / previous_peak
        return drawdown.min()
    
    def index_volatility(self, index_ticker='^GSPC', resample_time=False):
        cum_returns, returns = self.index_returns(index_ticker=index_ticker, resample_time=resample_time)
        returns = returns.mean() * 252
        return returns.std()
    
    def corr(self, yahooData=True, graph=True):      
        df_corr = self.portfolio_close_data.pct_change().corr() 
        data1 = df_corr.values
        
        if graph:
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
        return df_corr
                        
    def cumulative_returns_graph(self, resample_time=False, budget=1, index='^GSPC', index_show=True):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        plt.plot(self.cumulative_returns(resample_time=resample_time, budget=budget), label='Portfolio')
        if index_show:
            index_returns_data, returns = self.index_returns(index_ticker=index, resample_time=resample_time, budget=budget)
            plt.plot(index_returns_data, label=index)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Growth of $"+ str(budget) +" investment")
        ax1.xaxis_date()
        ax1.autoscale_view()
        ax1.legend(loc="upper left") 
        ax1.set_title("Portfolio cumulative returns data")
        plt.show()

    def monte_carlo(self):
        #Generate portfolios with allocations
        portfolios_allocations_df = mcs.generate_portfolios(expected_returns, covariance, settings.RiskFreeRate)
        portfolio_risk_return_ratio_df = portfolios_allocation_mapper.map_to_risk_return_ratios(portfolios_allocations_df)
        
        #Plot portfolios, print max sharpe portfolio & save data
        cp.plot_portfolios(portfolio_risk_return_ratio_df)
        max_sharpe_portfolio = mc.get_max_sharpe_ratio(portfolio_risk_return_ratio_df)['Portfolio']
        max_shape_ratio_allocations = portfolios_allocations_df[[ 'Symbol', max_sharpe_portfolio]]
        print(max_shape_ratio_allocations)

    def otimizer(self, numberOfPortfolios): 
              
        #portfolios_allocations_df = self.individual_returns()
        
        index = self.tickers + ['Annualized Return', 'Volatility', 'SharpeRatio']
        portfolios_allocations_df = pd.DataFrame({'index': index})
        
        portfolio_size = len(self.tickers)
               
        np.random.seed(0) #Adding equal allocation so I can assess how good/bad it is
        my_allocations = np.array(self.weights)
        portfolio_id = 'MyAllocationPortfolio'
        
        my_allocations_returns = self.annualized_return(my_allocations)
        my_allocations_volatility = self.volatility(my_allocations)
        my_allocations_sharpe_ratio = self.sharpe_ratio_annualized(my_allocations)

            
        portfolio_data = my_allocations        
        portfolio_data = np.append(portfolio_data, my_allocations_returns)
        portfolio_data = np.append(portfolio_data, my_allocations_volatility)
        portfolio_data = np.append(portfolio_data, my_allocations_sharpe_ratio)
                   
        #add data to the dataframe            
        portfolios_allocations_df[portfolio_id] = portfolio_data
        
        for i in range(numberOfPortfolios):
            portfolio_id = 'Portfolio_'+str(i)
            allocations = self.get_random_allocations(portfolio_size)
            portfolio = Portfolio(stocks=self.tickers, weights=allocations, start_date=self.start_date, end_date=self.end_date)
            
            allocations_returns =  portfolio.annualized_return(allocations)
            allocations_volatility = portfolio.volatility(allocations)
            allocations_sharpe_ratio = portfolio.sharpe_ratio_annualized(allocations)
            
            portfolio_data = allocations
            portfolio_data = np.append(portfolio_data,allocations_returns)
            portfolio_data = np.append(portfolio_data,allocations_volatility)
            portfolio_data = np.append(portfolio_data,allocations_sharpe_ratio)
            
            portfolios_allocations_df[portfolio_id] = portfolio_data
            #portfolios_allocations_df = pd.DataFrame(data=portfolio_data.flatten())
        
        portfolios_allocations_df.set_index('index', inplace=True)
        return portfolios_allocations_df
    
    def get_random_allocations(self, portfolio_size):
        allocations = np.random.rand(portfolio_size)
        allocations /= sum(allocations)
        return allocations
   
    def efficient_frontier(self, numberOfPortfolios, graph=False):
        
        df = self.otimizer(numberOfPortfolios)
        
        sdp_my, rp_my = df['MyAllocationPortfolio']['Volatility'], df['MyAllocationPortfolio']['Annualized Return']
               
        max_sharpe_idx = df.idxmax(axis=1)['SharpeRatio']
        sdp, rp = df[max_sharpe_idx]['Volatility'], df[max_sharpe_idx]['Annualized Return']
        max_sharpe_ratio = df[max_sharpe_idx]['SharpeRatio']
        max_sharpe_allocation = []
        for i in range(len(self.tickers)):
            max_sharpe_allocation.append(df[max_sharpe_idx][i])
              
        min_vol_idx = df.idxmin(axis=1)['Volatility']
        min_vol_sharperatio = df[min_vol_idx]['SharpeRatio']
        sdp_min, rp_min = df[min_vol_idx]['Volatility'], df[min_vol_idx]['Annualized Return']
        min_vol_allocation = []
        for i in range(len(self.tickers)):
            min_vol_allocation.append(df[min_vol_idx][i])
        
#        print ("-"*80)
#        print ("Sharpe Ratio Portfolio Allocation\n")
#        print ("Annualised Return:", round(self.annualized_return(),2))
#        print ("Annualised Volatility:", round(self.volatility(),2))
#        print ("Sharpe Ratio", round(self.sharpe_ratio_annualized(),2))
#        print ("\n")
#        print ("-"*80)
#        print ("Maximum Sharpe Ratio Portfolio Allocation\n")
#        print ("Annualised Return:", round(rp,2))
#        print ("Annualised Volatility:", round(sdp,2))
#        print ("Sharpe Ratio", round(max_sharpe_ratio,2))
#        print ("\n")
#        print (max_sharpe_allocation)
#        print ("-"*80)
#        print ("Minimum Volatility Portfolio Allocation\n")
#        print ("Annualised Return:", round(rp_min,2))
#        print ("Annualised Volatility:", round(sdp_min,2))
#        print ("Sharpe Ratio", round(min_vol_sharperatio,2))
#        print ("\n")
#        print (min_vol_allocation)
        
        results = {
                "Sharpe Ratio Portfolio Allocation": self.weights,
                "Annualised Return": round(self.annualized_return(),4),
                "Annualised Volatility": round(self.volatility(),4),
                "Sharpe Ratio": round(self.sharpe_ratio_annualized(),4),
                "MaxSharpeRatio": round(max_sharpe_ratio,2),
                "MaxSharpe Annualised Return": round(rp,4),
                "MaxSharpe Annualised Volatility": round(sdp,4),
                "MinVolatility Sharpe Ratio": round(min_vol_sharperatio,2),
                "MinVolatility Annualised Return": round(rp_min,4),
                "Min Annualised Volatility": round(sdp_min,4),
                "MaxSharpeAllocation": max_sharpe_allocation, 
                "MinVolAllocation": min_vol_allocation
        }
            
        if graph:
            
            plt.figure(figsize=(10, 7))
            
            x = df.loc['Volatility']
            y = df.loc['Annualized Return']
            z = df.loc['SharpeRatio']
            
            plt.scatter(x, y , c = z , cmap='viridis')
            plt.colorbar(label='Sharpe Ratio')
            plt.scatter(sdp_my, rp_my, marker='*', s=300, label='MyAllocation')
            plt.scatter(sdp, rp , marker='*', color='r', s=300, label='Maximum Sharpe ratio')
            plt.scatter(sdp_min, rp_min ,marker='*', color='g', s=300, label='Minimum volatility')
            plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
            plt.xlabel('Volatility')
            plt.ylabel('Return')
            plt.legend(labelspacing=0.8)
        
        return max_sharpe_allocation, min_vol_allocation, results
    
        
    def otimizedPortfolio(self, numberOfPortfolios, maxSharpRatio=True, minVolatility=False, index='^GSPC', index_show=True):
        MaxSharp_weights, MinVol_weights, results = self.efficient_frontier(numberOfPortfolios)
        tickers = self.tickers
        
        if maxSharpRatio:
            weights = MaxSharp_weights
        if minVolatility:
            weights = MinVol_weights
        
        efficientPortfolio = Portfolio(stocks=tickers, weights=weights, start_date=self.start_date, end_date=self.end_date)
        
        if index_show:
            efficientPortfolio.cumulative_returns_graph(index=index, index_show=index_show)
        return efficientPortfolio
        
    def gbm(self, portfolio=None, n_years=3, n_scenarios=1000, mu=None, sigma=None, steps_per_year=252, numberofPortfolios=100, maxSharpRatio=True, minVolatility=False, budget=settings.budget, graph=False):
        
        if portfolio is None:
            portfolio = self.otimizedPortfolio(numberofPortfolios, maxSharpRatio=maxSharpRatio, minVolatility=minVolatility, index_show=False)
        else:
            portfolio = portfolio
        
        if mu is None:
            mu = portfolio.annualized_return()
        if sigma is None:
            sigma = portfolio.volatility()
        
        dt = 1/steps_per_year
        n_steps = int(n_years * steps_per_year)
        rets_plus1 = np.random.normal(loc=(1+mu*dt), scale=(sigma*np.sqrt(dt)) , size=(n_steps+1, n_scenarios))
        rets_plus1[0] = 1
                
        gbm = budget*pd.DataFrame(rets_plus1).cumprod()
        
        if graph:
            terminal_wealth = gbm.iloc[-1]        
            tw_mean = terminal_wealth.mean()
            tw_median = terminal_wealth.median()
            
            ax = gbm.plot(legend=False, color='lightblue', alpha=0.5, linewidth=2, figsize=(12,5))
            
            ax.axhline(y=tw_mean, ls=':', color='indianred')
            ax.axhline(y=tw_median, ls=':', color='purple')           
            ax.axhline(y=budget, ls=':', color='black')
            ax.set_ylim(top=3000)
            
            ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9), xycoords='axes fraction', fontsize=24)
            ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85), xycoords='axes fraction', fontsize=24)
            
            ax.plot(marker='o', color='darkred', alpha='0.2')
            
        return gbm, portfolio
    
    def interactive_plot(self):
        interactive_plot = widgets.interactive(self.show_gbm, 
                                       n_scenarios=(10, 1000, 50), 
                                       mu=(0, 0.2, 0.01),
                                       sigma=(0, 0.3, 0.01)
                                      )
        output = interactive_plot.children[-1]
        output.layout.height = '350px'
        return interactive_plot
    
    def run_cppi(self, risky_r, safe_r=None, m=settings.multiplier, start=settings.budget, floor=settings.cppi_floor, RiskFree_Rate=settings.RiskFree_Rate):
        dates = risky_r.index
        risky_r = risky_r.pct_change()
        risky_r.loc[0] = 0
        
        n_steps = len(dates)
        account_value = start
        floor_value = start*floor
        
        safe_r = pd.DataFrame().reindex_like(risky_r)
        #safe_r.values[:] = riskfree_rate/12
        safe_r.values[:] = RiskFree_Rate/252
        
        account_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)
        
        #for step in range(n_steps):
        for step in range(n_steps):
            cushion = (account_value - floor_value) / account_value
            risky_w = m*cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1-risky_w
            risky_alloc = account_value*risky_w
            safe_alloc = account_value*safe_w
            
            account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
            
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value

        risky_wealth = start*(1+risky_r).cumprod()
        backtest_result = {
                "Wealth": account_history,
                "Risky Wealth": risky_wealth,
                "Risk Budget": cushion_history,
                "Risk Allocation": risky_w_history,
                "m": m,
                "start": start,
                "floor": floor,
                "risky_r": risky_r,
                "safe_r": safe_r
        }
        return backtest_result
    
    def show_cppi(self, portfolio=None, n_scenarios=1000, numberOfPortfolios=1000, m=settings.multiplier, floor=settings.cppi_floor, RiskFree_Rate=settings.RiskFree_Rate, y_max=100, start=settings.budget):
        
        
        risky_r, otimizedPortfolio = self.gbm(portfolio=portfolio, n_scenarios=n_scenarios, numberofPortfolios=numberOfPortfolios, graph=False)
        
        btr = self.run_cppi(risky_r = risky_r, RiskFree_Rate=RiskFree_Rate, m=m, start=start, floor=floor)
        wealth = btr["Wealth"]
               
        y_max = wealth.values.max()*y_max/100
        terminal_wealth = wealth.iloc[-1]
        
        tw_mean = terminal_wealth.mean()
        tw_median = terminal_wealth.median()
        
        failure_mask = np.less(terminal_wealth, start*floor)
        n_failures = failure_mask.sum()
        p_fail = n_failures / n_scenarios
        
        e_shortfall = np.dot(terminal_wealth*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
        
        fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]})
        plt.subplots_adjust(wspace=0.0)
        
        wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color='indianred')
        wealth_ax.axhline(y=start, ls=':', color='black')
        wealth_ax.axhline(y=start*floor, ls='--', color='red')
        wealth_ax.set_ylim(top=y_max)
        
        terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
        hist_ax.axhline(y=start, ls=':', color='black')
        hist_ax.axhline(y=tw_mean, ls=':', color='blue')
        hist_ax.axhline(y=tw_median, ls=':', color='purple')
        
        hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9), xycoords='axes fraction', fontsize=24)
        hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85), xycoords='axes fraction', fontsize=24)
        
        if (floor > 0.01):
            hist_ax.axhline(y=start*floor, ls='--', color='red', linewidth=3)
            hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)
        
    def gap(self, gap_years, numberOfPortfolios):
        data = self.portfolio_close_data
        data = data.dropna()
        start_year = data.index[0].year
        last_year = data.index[-1].year
        sharpe_ratio_all, sharpe_ration_otimizer, returns_all, returns_otimizer, volatility_all = [], [], [], [], []
        years = []
        
        a, b, results = self.efficient_frontier(numberOfPortfolios)
        
        while last_year - gap_years + 1 > start_year:
            portfolio = Portfolio(stocks=self.tickers, weights=self.weights, start_date=str(last_year - gap_years + 1)+'-01-01', end_date=str(last_year)+'-12-31')
            #a, b, results = portfolio.efficient_frontier(numberOfPortfolios)
            
            years.append(last_year)
            sharpe_ratio_all.append(portfolio.sharpe_ratio_annualized())
            returns_all.append(portfolio.annualized_return())
            volatility_all.append(portfolio.volatility())
#            sharpe_ratio_all.append(results['Sharpe Ratio'])
#            sharpe_ration_otimizer.append(results['MaxSharpeRatio'])
#            returns_all.append(round(results['Annualised Return'] * 100, 2))
#            returns_otimizer.append(round(results['MaxSharpe Annualised Return'] * 100, 2))
#            
            print ('Period: ' + str(last_year - gap_years + 1) + '-' + str(last_year) + ' -',
                   'SharpeRatio',
                   sharpe_ratio_all[-1],
                   'Annualised Return:',
                   str(round(returns_all[-1] * 100, 2)) + '%',
                   'Annualised Volatility:',
                   str(round(volatility_all[-1] * 100, 2)) + '%'
                  )
            
#            print ('Period: ' + str(last_year - gap_years + 1) + '-' + str(last_year) + ' -',
#                   'SharpeRatio',
#                   results['MaxSharpeRatio'],
#                   'Annualised Return:',
#                   str(round(results['MaxSharpe Annualised Return'] * 100, 2)) + '%',
#                   'Annualised Volatility:',
#                   str(round(results['MaxSharpe Annualised Volatility'] * 100, 2)) + '%',
#                   results['MaxSharpeAllocation']
#                  )
#            print ('Period: ' + str(last_year - gap_years + 1) + '-' + str(last_year) + ' -',
#                   'SharpeRatio',
#                   results['Sharpe Ratio'],
#                   'Annualised Return:',
#                   str(round(results['Annualised Return'] * 100, 2)) + '%',
#                   'Annualised Volatility:',
#                   str(round(results['Annualised Volatility'] * 100, 2)) + '%',
#                   results['Sharpe Ratio Portfolio Allocation']
#                  )
            
            last_year -= gap_years

        
        
        fig, (ax1, ax2) = plt.subplots(1,2, sharex=True)
        
        ax1.plot(years, returns_all, c='red', marker="o", label='Returns Allocation')
        ax1.plot(years, returns_otimizer, c='y', marker="o", label='Returns Allocation Otimizer')
        
        ax2.plot(years, sharpe_ratio_all, c='blue', marker="s", label='SharpeRatio Allocation')
        ax2.plot(years, sharpe_ration_otimizer, c='c', marker="s", label='SharpeRatio Otimizer')
                
        ax2.axhline(y=0, ls='--', color='red', linewidth=1.5)
        ax2.axhline(y=1, ls='--', color='yellow', linewidth=1.5)
        ax2.axhline(y=2, ls='--', color='green', linewidth=1.5)
        
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        fig.suptitle('Portfolio ' + str(gap_years) + ' Year(s)')
        
        ax1.grid()
        ax2.grid()
        plt.show()
        