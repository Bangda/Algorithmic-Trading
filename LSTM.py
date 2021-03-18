### Quantiacs Trend Following Trading System Example
# import necessary Packages below:

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
import math
import json
import os
import numpy
import pandas as pd

script_dir = os.path.dirname(__file__)
date = "observation_date" 

with open("economics_data.json") as json_file:
    feature_data = json.load(json_file)
    dfs = []
    idx = pd.date_range('2017-01-01', '2021-03-12')
    for key,value in feature_data.items():
        print("loading ", key)
        
        path = value['filepath']
        abs_file_path = os.path.join(script_dir, path)
        df = pd.read_excel(abs_file_path)
        df[date] = pd.to_datetime(df[date])
        mask = (df[date] > "2017-01-01") & (df[date] <= "2021-03-12")
        df = df.loc[mask]
        df.index = pd.DatetimeIndex(df[date])
        df = df.reindex(idx,  fill_value=numpy.nan)
        df = df.drop(columns=[date]).reset_index().rename(columns={'index':date}).fillna(method="ffill").fillna(method="bfill")
        dfs.append(df)
        # break
features = pd.concat(dfs, axis=1)
features = features.loc[:,~features.columns.duplicated()]

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    try:
        print(settings['counter'])
    except:
        settings['counter'] = 0
        settings['LSTM_regressor'] = dict()
        settings['sc'] = dict()

    nMarkets = CLOSE.shape[1]
    pos = numpy.zeros(nMarkets)
    print(DATE[0])
    count_in_cash = 0
    for market in range(1,nMarkets):
        print("Processing market, index: {}".format(market))

        close = CLOSE[:, market]
        DATE_ = DATE
        data = pd.DataFrame()
        print(len(close) == len(DATE_))
        data['CLOSE'] = close
        data['observation_date'] = DATE_
        data['observation_date'] = pd.to_datetime(data['observation_date'].astype(str))
        data = data.merge(features, on=date, how="left")
        print(data.head())

        # retrain the lSTM model every 100 days
        if settings['counter']%100==0:
            training_set = numpy.reshape(CLOSE[:,market], (CLOSE[:,market].shape[0], 1))
            print("training_set", training_set.shape)
            training_set = (training_set-numpy.insert(training_set, 0, 0, axis=0)[:-1,])/numpy.insert(training_set, 0, 0, axis=0)[:-1,]
            training_set = training_set[1:,]
            print(training_set.shape)

            sc = MinMaxScaler(feature_range = (0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            settings['sc'][str(market)] = sc 

            
            X_train = []
            y_train = []
            
            for i in range(30, training_set.shape[0]):
                X_train.append(training_set_scaled[i-30:i,0])
                y_train.append(training_set_scaled[i,0])
            
            X_train, y_train = numpy.array(X_train), numpy.array(y_train)
            print("len of X_train", len(X_train))
            X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            print(X_train.shape)
            

            # LSTM
            print("Re-training LSTM!")
            regressor = Sequential()

            regressor.add(LSTM(units = 25, return_sequences = True, input_shape = (X_train.shape[1], 1)))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units = 20, return_sequences = True))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units = 5))
            regressor.add(Dropout(0.1))

            regressor.add(Dense(units = 1))

            regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')

            regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)
            
            settings['LSTM_regressor'][str(market)] = regressor
            
            print("Completed re-training!")
        else:
            print("Deploying existing LSTM!")
            sc = settings['sc'][str(market)]
            regressor = settings['LSTM_regressor'][str(market)]
        

        X_pred = []
        pred_set = numpy.reshape(CLOSE[:,market], (CLOSE[:,market].shape[0], 1))


        pred_set = (pred_set - numpy.insert(pred_set, 0, 0, axis=0)[:-1,])/numpy.insert(pred_set, 0, 0, axis=0)[:-1,]
        pred_set = pred_set[1:,]
        pred_set_scaled = sc.fit_transform(pred_set)
        X_pred.append(pred_set_scaled[-30:,0])
        X_pred = numpy.array(X_pred)
        X_pred = numpy.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        predicted_stock_price = regressor.predict(X_pred)
        print(predicted_stock_price)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        err_term = 0.005 #%[0.005:0.001:0.008]#
        if predicted_stock_price[0][0] > 0:
            pos[market] = 1
            # print('LONG')


        elif predicted_stock_price[0][0] + err_term < 0:
            pos[market] = -1
            # print('SHORT') 

        else:
            pos[0] = pos[0] + 1 

        print('*' * 100)
    
    settings['counter'] = settings['counter'] + 1

    return pos, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings = {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets'] = ['CASH', 'F_AD'] 
    # settings['markets'] = ['CASH', 'F_AD', 'F_AE', 'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA',
    #                        'F_CC', 'F_CD', 'F_CF', 'F_CL', 'F_CT', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_DZ', 'F_EB',
    #                        'F_EC', 'F_ED', 'F_ES', 'F_F',  'F_FB', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY',
    #                        'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_HG', 'F_HO', 'F_HP', 'F_JY', 'F_KC', 'F_LB', 'F_LC',
    #                        'F_LN', 'F_LQ', 'F_LR', 'F_LU', 'F_LX', 'F_MD', 'F_MP', 'F_ND', 'F_NG', 'F_NQ', 'F_NR',
    #                        'F_NY', 'F_O',  'F_OJ', 'F_PA', 'F_PL', 'F_PQ', 'F_RB', 'F_RF', 'F_RP', 'F_RR', 'F_RU',
    #                        'F_RY', 'F_S',  'F_SB', 'F_SF', 'F_SH', 'F_SI', 'F_SM', 'F_SS', 'F_SX', 'F_TR', 'F_TU',
    #                        'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VT', 'F_VW', 'F_VX',  'F_W', 'F_XX', 'F_YM',
    #                        'F_ZQ']

    settings['beginInSample'] = '20190123'
    settings['endInSample'] = '20210305'
    settings['lookback'] = 504
    settings['budget'] = 10**6
    settings['slippage'] = 0.05

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    print(' ')
    print('Stats:')
    print(results['stats'])
    print('Returns:')
    print(results['returns'])
    # optimize(__file__)