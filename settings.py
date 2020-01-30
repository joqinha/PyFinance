#Settings

class settings:
    FMP_data = 'fmp_data'
    Yahoo_Data_Path = 'yahoo_data'
    info_file_path = 'tickers_info.csv'
    yahoo_close = 'Adj Close'
    yahoo_date = 'Date'
    FMP_close = 'close'
    FMP_date = 'date'
    date0 = '1999-01-01'
    RiskFree_Rate=0.02
    budget = 1000
    cppi_floor = 0.9
    multiplier = 4
    no_values = ['BKR','BRK.B','BF.B','PEAK','NLOK', 'OMAM']
    
    exchange_year = 2.5
    etf_transaction_fee = 2
    etf_fee = 0.038
    portugal_tax = 0.28
    
    etf_tickers = {
                        '^GSPC':
                        {
                         'Description': 'S&P 500 Indice'},

                        ##########-----WORLD-----##########
                        'SWRD.AS':
                        {
                        'Description': 'SPDR MSCI World UCITS ETF',
                        'Exchange': 'XAMS',
                        'Type': 'Acc',
                        'Degiro Free': False,
                        'Fee': 0.12},
                                
                        'SPPW.DE':
                        {
                        'Description': 'SPDR MSCI World UCITS ETF',
                        'Exchange': 'XETR',
                        'Type': 'Acc',
                        'Degiro Free': False,
                        'Fee': 0.12},

                        'IWDA.AS':
                         {'Description': 'iShares Core MSCI World UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': True, 
                         'Fee': 0.20},
                                
                        'EUNL.DE':
                         {'Description': 'iShares Core MSCI World UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.20},

                        'CSPX.AS':
                         {'Description': 'iShares Core S&P 500 UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        'SXR8.DE':
                         {'Description': 'iShares Core S&P 500 UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        
                        'IJPA.AS':
                         {'Description': 'iShares Core MSCI Japan IMI UCITS ETF USD', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.15},
                        'EUNN.DE':
                         {'Description': 'iShares Core MSCI Japan IMI UCITS ETF USD', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.15},
                        
                        'IKRA.AS':
                         {'Description': 'iShares MSCI Korea UCITS ETF USD', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True,
                         'Fee': 0.74},
                        'IQQK.DE':
                         {'Description': 'iShares MSCI Korea UCITS ETF USD', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False,
                         'Fee': 0.74
                         },

                        'SXRG.DE':
                         {'Description': 'iShares MSCI USA Small Cap UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.43},
                        
                        'IUSN.DE':
                         {'Description': 'iShares MSCI World Small Cap UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.35},
                        
                        'IQQ0.DE':
                         {'Description': 'iShares Edge MSCI World Minimum Volatility UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.30},
                        
                        'QDVE.DE':
                         {'Description': 'iShares S&P 500 Information Technology Sector UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.15},
                        'QDVG.DE':
                         {'Description': 'iShares S&P 500 Health Care Sector UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.15},
                                        
                        
                        ##########-----EUROPE-----##########
                        'CSX5.AS':
                         {'Description': 'iShares Core EURO STOXX 50 ETF EUR', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.10},
                        'SXRT.DE':
                         {'Description': 'iShares Core EURO STOXX 50 ETF EUR', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.10},
                        'CEMU.AS':
                         {'Description': 'iShares Core MSCI EMU UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.12},
                        'SXR7.DE':
                         {'Description': 'iShares Core MSCI EMU UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.12},
                        'EUNA.AS':
                         {'Description': 'iShares STOXX Europe 50 UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True, 
                         'Fee': 0.35},
                        'EUN1.DE':
                         {'Description': 'iShares STOXX Europe 50 UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.35},
                        
                        'SXRJ.DE':
                         {'Description': 'iShares MSCI EMU Small Cap UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.58},
                        
                        ##########-----EMERGING MARKETS-----##########
                        'EMIM.AS':
                         {'Description': 'iShares Core MSCI EM IMI UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.18},
                        'IS3N.DE':
                         {'Description': 'iShares Core MSCI EM IMI UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.18},
                        
                        'IBZL.AS':
                         {'Description': 'iShares MSCI Brazil UCITS ETF Eur', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True, 
                         'Fee': 0.74},
                        'IQQB.DE':
                         {'Description': 'iShares MSCI Brazil UCITS ETF Eur', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False,
                         'Fee': 0.74},
                        
                        'CEBL.DE':
                         {'Description': 'iShares MSCI EM Asia UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False,
                         'Fee': 0.65},
                        
                        'IQQC.DE':
                         {'Description': 'iShares China Large Cap UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.74},
                        'FXC.AS':
                         {'Description': 'iShares China Large Cap UCITS ETF', 
                         'Exchange': 'XAMS',
                         'Type': 'Dist', 
                         'Degiro Free': True,
                         'Fee': 0.74},
                        
                        'EUNI.DE':
                         {'Description': 'iShares MSCI EM Small Cap UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.74},

                        
                        ##########-----BONDS-----##########
                        'ICOV.AS':
                         {'Description': 'iShares € Covered Bond UCITS ETF',
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True,
                         'Fee': 0.20},
                        'IUS6.DE':
                         {'Description': 'iShares € Covered Bond UCITS ETF', 
                         'Exchange': 'XETR',
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.20},
                        
                        'LQDA.AS':
                         {'Description': 'iShares $ Corp Bond UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True, 
                         'Fee': 0.20},
                        'IBCD.DE':
                         {'Description': 'iShares $ Corp Bond UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.20},
                        
                        
                        #### -----0.07%-----####
                        'IBCC.DE':
                         {'Description': 'iShares $ Treasury Bond 0-1yr UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        'IBTS.AS':
                         {'Description': 'iShares $ Treasury Bond 1-3yr UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True,
                         'Fee': 0.07},
                        'IUSU.DE':
                         {'Description': 'iShares $ Treasury Bond 1-3yr UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        'CBU7.AS':
                         {'Description': 'iShares $ Treasury Bond 3-7yr UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        'SXRL.DE':
                         {'Description': 'iShares $ Treasury Bond 3-7yr UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Acc', 
                         'Degiro Free': False, 
                         'Fee': 0.07},
                        'BTMA.AS':
                         {'Description': 'iShares $ Treasury Bond 7-10yr UCITS ETF', 
                         'Exchange': 'XAMS', 
                         'Type': 'Dist', 
                         'Degiro Free': True, 
                         'Fee': 0.07},
                        'IUSM.DE':
                         {'Description': 'iShares $ Treasury Bond 7-10yr UCITS ETF', 
                         'Exchange': 'XETR', 
                         'Type': 'Dist', 
                         'Degiro Free': False, 
                         'Fee': 0.07}
                   }
    yahoo_drop_columns = ['index', 'Open', 'High', 'Low', 'Close', 'Volume']
    fmp_drop_columns = ['index','open','high','low','volume','unadjustedVolume','change','changePercent','vwap','label','changeOverTime']