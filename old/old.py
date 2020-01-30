#OLD

#    if reload_sp500 or not os.path.isfile('sp500tickers.pickle'):
#        tickers = save_sp500_symbols()
#    else:
#        with open("sp500tickers.pickle", "rb") as f:
#            tickers = pickle.load(f)
#    if not os.path.exists('stock_dfs'):
#        os.makedirs('stock_dfs')

#def asset_frontier():
#    n_points = 20
#    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
#    return weights
        
##################################################
        
        
#def get_ticker_info(reload_sp500=False):
#    with open("sp500tickers.pickle", "rb") as f:
#        tickers = pickle.load(f)
#        
#    for ticker in tickers:
#        # just in case your connection breaks, we'd like to save our progress!
#        if not os.path.exists('stock_dfs/{}_info.csv'.format(ticker)) and ticker not in ['BKR','BRK.B','BF.B','PEAK','NLOK']:
#            print (ticker)
#            url_profile = 'https://financialmodelingprep.com/api/v3/company/profile/'
#            url = url_profile + ticker
#            resp = requests.get(url=url)
#            data = resp.json() # Check the JSON Response Content documentation below
#            df = pd.DataFrame(data)
#
#            df.to_csv('stock_dfs/{}_info.csv'.format(ticker))
#        else:
#            print('Already have {}'.format(ticker))
        
#def compile_data():
#    with open("sp500tickers.pickle", "rb") as f:
#        tickers = pickle.load(f)
#
#    main_df = pd.DataFrame()
#
#    for count, ticker in enumerate(tickers):
#        if ticker not in ['BKR','BRK.B','BF.B','PEAK','NLOK']:
#            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
#            df.set_index('date', inplace=True)
#            
#            df.rename(columns={'close': ticker}, inplace=True)
#            #df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
#            df.drop(['index','open','high','low','volume','unadjustedVolume','change','changePercent','vwap','label','changeOverTime'], 1, inplace=True)
#            
#            if main_df.empty:
#                main_df = df
#            else:
#                main_df = main_df.join(df, how='outer')
#                    
#            if count % 10 == 0:
#                print(count)
#            print(main_df.head())
#            main_df.to_csv('sp500_joined_closes.csv')
#            
#            
#
#
#def visualize_data():
#    df = pd.read_csv('sp500_joined_closes.csv')
#    #df_corr = df.corr()
#    df.set_index('date', inplace=True) 
#    df_corr = df.pct_change().corr() 
#    print(df_corr.head())
#    df_corr.to_csv('sp500corr.csv')
#    data1 = df_corr.values
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#
#    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
#    fig1.colorbar(heatmap1)
#
#    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
#    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
#    ax1.invert_yaxis()
#    ax1.xaxis.tick_top()
#    column_labels = df_corr.columns
#    row_labels = df_corr.index
#    ax1.set_xticklabels(column_labels)
#    ax1.set_yticklabels(row_labels)
#    plt.xticks(rotation=90)
#    heatmap1.set_clim(-1, 1)
#    plt.tight_layout()
#    plt.show()
#    
#def process_data_for_labels(ticker):
#    hm_days = 7
#    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
#    tickers = df.columns.values.tolist()
#    df.fillna(0, inplace=True)
#
#    for i in range(1,hm_days+1):
#        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
#
#    df.fillna(0, inplace=True)
#    return tickers, df, hm_days
#
#def buy_sell_hold(*args):
#    cols = [c for c in args]
#    requirement = 0.02
#    for col in cols:
#        if col > requirement:
#            return 1
#        if col < -requirement:
#            return -1
#    return 0
#
#
#def extract_featuresets(ticker):
#    tickers, df, hm_days = process_data_for_labels(ticker)
#        
#    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
#                                              *[df["{}_{}d".format(ticker, i)] for i in range(1, hm_days+1)]))
#    
#    vals = df['{}_target'.format(ticker)].values.tolist()
#    str_vals = [str(i) for i in vals]
#    print('Data spread:',Counter(str_vals))
#    
#    df.fillna(0, inplace=True)
#    df = df.replace([np.inf, -np.inf], np.nan)
#    df.dropna(inplace=True)
#    
#    df_vals = df[[ticker for ticker in tickers]].pct_change()
#    df_vals = df_vals.replace([np.inf, -np.inf], 0)
#    df_vals.fillna(0, inplace=True)
#    
#    X = df_vals.values
#    y = df['{}_target'.format(ticker)].values
#
#    return X,y,df
#def do_ml(ticker):
#    X, y, df = extract_featuresets(ticker)
#
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
#    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
#                            ('knn', neighbors.KNeighborsClassifier()),
#                            ('rfor', RandomForestClassifier())])
#
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_test, y_test)
#    print('accuracy:', confidence)
#    predictions = clf.predict(X_test)
#    print('predicted class counts:', Counter(predictions))
#    print()
#    print()
#    return confidence
#        
#def graph(ticker,*ma):
#    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)
#    for x in ma:
#        df[str(x)+'ma'] = df['close'].rolling(window=x, min_periods=0).mean()
#    
#    
#    df_ohlc = df['close'].resample('10D').ohlc()
#    df_volume = df['volume'].resample('10D').sum()
#    df_ohlc.reset_index(inplace=True)
#    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)
#    
#    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
#    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
#    ax1.xaxis_date()
#    
#    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
#    
#    for x in ma:
#        ax1.plot(df.index, df[str(x)+'ma'], label=str(x)+'ma')
#        
#    ax1.legend(loc="upper left")
#    
#    ax1.set_title(ticker)
#    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
#    plt.show()
#    
#    
#returns = prices.pct_change()
# 
## mean daily return and covariance of daily returns
#mean_daily_returns = returns.mean()
#cov_matrix = returns.cov()
# 
## portfolio weights
#weights = np.asarray([0.5,0.5])
# 
#portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
#portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
#
#print("Expected annualised return: " + str(portfolio_return))
#print("Volatility: " + str(portfolio_std_dev))

#def get_info_old(index='sp500'):
#    
#    info_file_path = index + '_info.csv'
#    
#    no_values = ['BKR','BRK.B','BF.B','PEAK','NLOK']
#    
#    with open("sp500tickers.pickle", "rb") as f:
#        tickers = pickle.load(f)
#        
#    main_df = pd.DataFrame()
#    
#    no_info=[]
#    
#    for ticker in tickers:
#        if ticker not in no_values:
#            print('Getting info data for {}'.format(ticker))
#            url_profile = 'https://financialmodelingprep.com/api/v3/company/profile/'
#            url = url_profile + ticker
#            resp = requests.get(url=url)
#            data = resp.json()
#            
#            df = pd.DataFrame(data)
#            if not df.empty:
#                df.drop(['beta', 'changes', 'changesPercentage', 'image', 'lastDiv', 'mktCap', 'price', 'range', 'volAvg'], inplace=True)            
#                #df.reset_index(inplace=True)
#                #df.set_index('index', inplace=True)
#                main_df = pd.concat([main_df, df]) 
#            else:
#                no_info.append(ticker)
#                
#    main_df.to_csv(info_file_path)
#    if no_info :
#        print('No info data for ',no_info)
#          

#def save_sp500_symbols():
#    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#    soup = bs.BeautifulSoup(resp.text, 'lxml')
#    table = soup.find('table', {'class': 'wikitable sortable'})
#    tickers = []
#    for row in table.findAll('tr')[1:]:
#        ticker = row.findAll('td')[0].text
#        tickers.append(ticker[:-1])
#    with open("sp500tickers.pickle", "wb") as f:
#        pickle.dump(tickers, f)
#    return tickers