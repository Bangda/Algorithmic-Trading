### Quantiacs Trading System Template


# import necessary Packages below:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    '''Define your trading system here.

    See the example trading system for a starting point.

    The function name "myTradingSystem" should no be changed. We evaluate this function on our server.

    Your system should return a normalized set of weights for the markets you have defined in settings['markets']. '''
    try:
        # print(settings['history'])
        type(settings['history']) == dict
    except:
        settings['history'] = {}
    

    nMarkets = CLOSE.shape[1]

    pos = np.zeros((1, nMarkets), dtype=np.float)

    for market in range(nMarkets):
        try:
        # print(settings['history'])
            print(settings['history'][market])
        except:
            settings['history'][market] = []

        close = CLOSE[:, market]
        data = pd.DataFrame()
        data['close'] = close
        close = data['close']

        periodLonger = 200 #%[100:10:200]#
        periodShorter = 10 #%[40:10:100]#

        # Get the difference in price from previous step
        delta = close.diff()
        # Get rid of the first row, which is NaN since it did not have a previous 
        # row to calculate the differences
        delta = delta[1:] 

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the SMA
        roll_up2 = up.rolling(periodShorter).mean()
        roll_down2 = down.abs().rolling(periodShorter).mean()

        # Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))
        # print("RSI for market [", market, "] is ", RSI2.values[-30:-1])

        # Calculate 200 day Simple Moving Average (SMA)
        smaLongerPeriod = np.nansum(CLOSE[-periodLonger-1:-1, market], axis=0)/periodLonger
        smaShorterPeriod = RSI2.values[-2]
        RSI_10days = RSI2.values[-11:-1]

        currentPrice = CLOSE[-1, market]

        # if uptrend and RSI < 30, long
        if currentPrice >= smaLongerPeriod and smaShorterPeriod < 30 and not np.isnan(smaShorterPeriod):
            pos[0, market] = 1
            settings['history'][market].append(1)
            print('long, previous day uptrend == ', currentPrice >= smaLongerPeriod, "previous day SRI = ", smaShorterPeriod)

        # if RSI > 40, sell
        elif smaShorterPeriod > 40:
            pos[0, market] = 0
            settings['history'][market].append(0)
            print('Sell, previous day uptrend == ', currentPrice >= smaLongerPeriod, "previous day SRI = ", smaShorterPeriod)

        # exit the market if hold the future for 10 days
        elif len(settings['history'][market]) > 10 and sum(settings['history'][market][-10:]) == 10:
            pos[0, market] = 0
            settings['history'][market].append(0)
            print('Sell because holding for more than 10 days, previous day uptrend == ', currentPrice >= smaLongerPeriod, "previous day SRI = ", smaShorterPeriod)

        # if not uptrend and not holding future, don't do anything
        elif currentPrice < smaLongerPeriod and smaShorterPeriod < 40 and len(settings['history'][market]) > 0 and settings['history'][market][-1] == 0:
            pos[0, market] = 0
            settings['history'][market].append(0)
            print('Hold cash, previous day uptrend == ', currentPrice >= smaLongerPeriod, "previous day SRI = ", smaShorterPeriod)

        # if not uptrend but holding the future. RSI < 40, hold future
        elif currentPrice < smaLongerPeriod and smaShorterPeriod < 40 and len(settings['history'][market]) > 0 and settings['history'][market][-1] == 1:
            pos[0, market] = 1
            settings['history'][market].append(1)
            print('Keep future, previous day uptrend == ', currentPrice >= smaLongerPeriod, "previous day SRI = ", smaShorterPeriod)

        else:
            pos[0, market] == 1
            settings['history'][market].append(1)

    
    weights = pos/np.nansum(abs(pos))
    print('Final position, ', pos)
    print('total number of trade, ', sum(settings['history'][market]))
    return weights, settings


def mySettings():
    '''Define your market list and other settings here.

    The function name "mySettings" should not be changed.

    Default settings are shown below.'''

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
    settings['markets'] = ['CASH','F_AD', 'F_AE'] #, 'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA']
    # settings['markets'] = ['CASH', 'F_AD', 'F_AE', 'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA',
    #                        'F_CC', 'F_CD', 'F_CF', 'F_CL', 'F_CT', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_DZ', 'F_EB',
    #                        'F_EC', 'F_ED', 'F_ES', 'F_F',  'F_FB', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY',
    #                        'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_HG', 'F_HO', 'F_HP', 'F_JY', 'F_KC', 'F_LB', 'F_LC',
    #                        'F_LN', 'F_LQ', 'F_LR', 'F_LU', 'F_LX', 'F_MD', 'F_MP', 'F_ND', 'F_NG', 'F_NQ', 'F_NR',
    #                        'F_NY', 'F_O',  'F_OJ', 'F_PA', 'F_PL', 'F_PQ', 'F_RB', 'F_RF', 'F_RP', 'F_RR', 'F_RU',
    #                        'F_RY', 'F_S',  'F_SB', 'F_SF', 'F_SH', 'F_SI', 'F_SM', 'F_SS', 'F_SX', 'F_TR', 'F_TU',
    #                        'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VT', 'F_VW', 'F_VX',  'F_W', 'F_XX', 'F_YM',
    #                        'F_ZQ']

    settings['lookback'] = 201
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20180101'
    settings['endInSample'] = '20201231'

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    print(' ')
    print('History:')
    print(results['history'])
    print('Returns:')
    print(results['returns'])
