import numpy as np
import pandas as pd

def process_mvo_df(df, stock_dimension):
    """
    
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)[['date','tic','close']]
    fst = df
    fst = fst.iloc[0:stock_dimension, :]
    tic = fst['tic'].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0
    
    for i in range(df.shape[0] // stock_dimension):
        n = df
        n = n.iloc[i * stock_dimension: (i+1) * stock_dimension, :]
        date= n['date'][i*stock_dimension]
        mvo.loc[date] = n['close'].tolist()

    return mvo

def StockReturnsComputing(stockPrice, rows, columns):
    """
    This calculates ?
    """
    import numpy as np 
    
    stockReturn = np.zeros([rows-1, columns])
    for j in range(columns): #j: assets
        for i in range(rows-1): #j: daily prices
            stockReturn[i,j] = ((stockPrice[i+1, j] - stockPrice[i, j])/stockPrice[i,j])*100

    return stockReturn

from pypfopt.efficient_frontier import EfficientFrontier
def compute_mvo_weights(meanReturns, covReturns, stock_dimension):
    ef_mean= EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean=ef_mean.clean_weights()
    mvo_weights=np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])

    return mvo_weights