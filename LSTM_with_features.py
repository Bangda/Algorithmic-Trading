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
features = pd.concat(dfs, axis=1)
features = features.loc[:,~features.columns.duplicated()]

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    try:
        print(settings['counter'])
    except:
        settings['counter'] = 0
        settings['LSTM_regressor'] = dict()
        settings['scaler_close'] = dict()
        settings['scaler_other'] = dict()
        settings['predicted_price'] = dict()
        settings['actual_price'] = dict()


    nMarkets = CLOSE.shape[1]
    pos = numpy.zeros(nMarkets)
    count_in_cash = 0
    for market in range(1,nMarkets):
        print("Processing market, index: {}".format(market))

        if settings['counter']%300==0:
            close = CLOSE[:, market]
            DATE_ = DATE
            data = pd.DataFrame()
            # print(len(close) == len(DATE_))
            data['CLOSE'] = close
            data['CLOSE'] = numpy.log(data['CLOSE']/data['CLOSE'].shift(1))
            data['CLOSE'] = data['CLOSE']

            data['observation_date'] = DATE_
            data['observation_date'] = pd.to_datetime(data['observation_date'].astype(str))
            data = data.merge(features, on=date, how="left")
            data.drop(columns=['observation_date'], inplace=True)


            close_col = data[['CLOSE']]
            other_col = data.drop(columns=['CLOSE'])
            values_close = close_col.values
            values_other = other_col.values

            # ensure all data is float
            values_close = values_close.astype('float32')
            values_other = values_other.astype('float32')

            # normalize features
            scaler_close = MinMaxScaler(feature_range=(0, 1))
            scaler_other = MinMaxScaler(feature_range=(0, 1))

            scaled_close = scaler_close.fit_transform(values_close)
            scaled_other = scaler_other.fit_transform(values_other)
            settings['scaler_close'][str(market)] = scaler_close 
            settings['scaler_other'][str(market)] = scaler_other

            scaled = numpy.concatenate([scaled_close,scaled_other], axis=1)

            # frame as supervised learning
            reframed = series_to_supervised(scaled, 3,2,dropnan=False)
            # remove next day features
            reframed.drop(reframed.columns[[-1,-2,-3,-5,-6,-7]], axis=1, inplace=True)

            # save the last 30 row for prediction of next day price
            pred_set = reframed.iloc[-15:].values[:, :-1]

            # leave the rest for training, remove NA
            training_set = reframed.iloc[:-1].dropna()

            # create training features and labels
            X_train = []
            y_train = []

            for i in range(15, training_set.shape[0]):
                X_train.append(training_set.values[i-15:i,:-1])
                y_train.append(training_set.values[i,-1])

            X_train, y_train = numpy.array(X_train), numpy.array(y_train)
            X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 13))

            regressor = Sequential()

            regressor.add(LSTM(units = 25, return_sequences = True, input_shape = (X_train.shape[1], 13)))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units = 20, return_sequences = True))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units = 5))
            regressor.add(Dropout(0.1))

            regressor.add(Dense(units = 1))

            regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')

            regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)
            settings['LSTM_regressor'][str(market)] = regressor
            settings['predicted_price'][str(market)] = []
            settings['actual_price'][str(market)] = []


            print("Completed re-training!")
        else:
            print("Deploying existing LSTM!")
            scaler_close = settings['scaler_close'][str(market)]
            scaler_other = settings['scaler_other'][str(market)]
            regressor = settings['LSTM_regressor'][str(market)]

        # making prediction
        close = CLOSE[:, market]
        DATE_ = DATE
        data = pd.DataFrame()
        data['CLOSE'] = close
        data['observation_date'] = DATE_
        data['observation_date'] = pd.to_datetime(data['observation_date'].astype(str))
        data = data.merge(features, on=date, how="left")
        data.drop(columns=['observation_date'], inplace=True)

        close_col = data[['CLOSE']]
        other_col = data.drop(columns=['CLOSE'])
        values_close = close_col.values
        values_other = other_col.values

        # ensure all data is float
        values_close = values_close.astype('float32')
        values_other = values_other.astype('float32')

        # normalize features
        scaled_close = scaler_close.transform(values_close)
        scaled_other = scaler_other.transform(values_other)

        scaled = numpy.concatenate([scaled_close,scaled_other], axis=1)

        # frame as supervised learning
        reframed = series_to_supervised(scaled, 3,2,dropnan=False)
        # remove next day features
        reframed.drop(reframed.columns[[-1,-2,-3,-5,-6,-7]], axis=1, inplace=True)

        # save the last 30 row for prediction of next day price
        pred_set = reframed.iloc[-30:].values[:, :-1]

        pred_set = numpy.reshape(pred_set, (1, pred_set.shape[0], 13))
        predicted_stock_price = regressor.predict(pred_set)
        predicted_stock_price = scaler_close.inverse_transform(predicted_stock_price)[0][0]
        current_value = CLOSE[-1,market]
        if settings['predicted_price'][str(market)] is not None and  type(settings['predicted_price'][str(market)]) == list:

            settings['predicted_price'][str(market)].append(predicted_stock_price)
        else:
            settings['predicted_price'][str(market)] = []
            settings['predicted_price'][str(market)].append(predicted_stock_price)

        if settings['actual_price'][str(market)] is not None and type(settings['actual_price'][str(market)]) == list:

            settings['actual_price'][str(market)].append(predicted_stock_price)
        else:
            settings['actual_price'][str(market)] = []
            settings['actual_price'][str(market)].append(predicted_stock_price)

        print("Predicted Price: ", predicted_stock_price)
        print("Current Price  : ", current_value)
        # compare predicted value for the next time period to fitted value of the current time period
        if predicted_stock_price > 0:
            pos[market] = 1
            print('Long')
        elif predicted_stock_price < -0.005:
            pos[market] = -1
            print('Short')
        else:
            pos[0] = pos[0] + 1
            print('Hold CASH')
    
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
    settings['markets'] = ['CASH', 'F_AD', 'F_AE', 'F_AH'] 
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
    # print(' ')
    print('Stats:')
    print(results['stats'])
    print('Returns:')
    print(results['returns'])
    optimize(__file__)