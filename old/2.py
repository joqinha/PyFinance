import requests
import pandas as pd
import matplotlib.pyplot as plt

stocks = ['AAPL']

stocks = ['AAPL', 
          'MSFT', 
          'BABA',
          'AERI',
          'ADSK',
          'WIFI',
          'CBS',
          'CNC',
          'XEC',
          'CXO',
          'DD',
          'DBX',
          'FANG',
          'DG',
          'LLY',
          'FFIV',
          'HAL',
          'INGR',
          'KSS',            
          'KR',
          'MRO',
          'MYL',
          'NTNX',
          'PEP',
          'RL',
          'RVNC',
          'CRM',
          'TPR',
          'UNH',
          'VZ',
          'VIAB',
          'YEXT',
          'MSFT'
]

ncols = 4;
nrows = int(len(stocks) / ncols)
if len(stocks)%ncols:
    nrows+=1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

i=0;
j=0;
for stock in stocks:
    url = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + stock + '?timeseries=365'
    resp = requests.get(url=url)
    data = resp.json() # Check the JSON Response Content documentation below
    
    df = pd.DataFrame(data['historical'])
    df = df.iloc[-30:]
    df.date = df.date.map(lambda x: x.lstrip('2019')).map(lambda y: y.lstrip('-'))    
    df.plot(x='date', y='close', figsize=(15, 15), ax=axes[j,i], sharex=True, legend=False, grid=True)
    axes[j,i].set_title(stock)
    i+=1;
    if i==ncols:
        i=0;
        j+=1;
