####### Covid Period ########
testStartDate = '01-01-2020'
testEndDate = '03-23-2020'
year = '2020'
filename =  './Data/^BVSP - 2020.csv'
############################

import pandas as pd
import numpy as np
import talib as talib   ###pip install talib-binary  ###https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)

def runModel(modelName, filename, dataFrameResult):
    print('-------------',modelName,'--------------')
    df = loadData(filename)
    df, X_train, X_test, y_train, y_test, X_train_norm, Y_train_norm, X_test_norm, y_test_norm, scaler_X_train, scaler_y_train = manipulateData(df, normalize=True)
    df, result, dict_error = fitAndPredict(modelName,df, X_train, X_test, y_train, y_test, X_train_norm, Y_train_norm, X_test_norm, y_test_norm, scaler_X_train, scaler_y_train)
    mse, rmse = calculateError(y_test, result['Pred_'+modelName])
    df = movement(df, modelName)
    print('Test Error (MSE, RMSE):', mse, rmse)
    hits =  df['Hit_Pred_'+modelName].sum()
    print(modelName,'. Hits: ',hits)
    
    print('Printing graph ...')
    graph = df[result.index[0]:]
    fig = graph[['Close_Tomorrow', 'Pred_'+modelName]][:testEndDate].plot(figsize=(20, 5)).get_figure()
    fig.savefig('./Output/'+modelName+'_'+testStartDate+'_'+testEndDate+'.pdf')
    
    print('Calculating ROI... ')
    import backtrader as bt
    from Trader_Strategy_ML import MLStrategy
    import Trader as trade
        
    data = bt.feeds.YahooFinanceCSVData(
            dataname=filename, 
            fromdate=pd.to_datetime(testStartDate),    #cst.period_03_train['startTest'],
            todate=pd.to_datetime(testEndDate))      #cst.period_03_train['endTest'])
    
    df_bt = pd.DataFrame()
    df_bt['Date'] = df[testStartDate:testEndDate].index
    df_bt['pred'] = np.array(df['Movement_Pred_'+modelName][testStartDate:testEndDate])df_bt = df_bt.set_index('Date')
    
    print('ML Strat\n')
    report = trade.createRunCerebro(data, MLStrategy, 'ML Trad'+modelName, 'ML Trad'+modelName+'_'+testStartDate+'_'+testEndDate , df_bt)    ##!! AJUSTAR
    print(report)
    
    print('------------- ----------------- --------------')

    dict_result = {'model': modelName}
    print(dict_result, dict_error)
    dict_result.update(dict_error)
    dict_result.update({'hits': hits})
    dict_result.update(report)
    dataFrameResult = dataFrameResult.append(dict_result, ignore_index = True)
    return dataFrameResult 

def loadData(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df[['Open']].plot(figsize=(15,3))
    
    return df #until 01-01-2021

def manipulateData(df, normalize = False):    
    from sklearn.model_selection import train_test_split
    df['Close_Tomorrow'] = df['Close'].shift(-1)
    if 'Adj Close' in df.columns: df = df.drop(["Adj Close"], axis=1)
    df = df[:-1] 
    df = df.dropna()
    
    
    #### techinical indicators ####
    #Technical Indicators - Tendency
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    #MA 10 CLOSE
    df['MA10'] = df['Close'].rolling(10).mean()
    #MA 20 CLOSE
    df['MA20'] = df['Close'].rolling(20).mean()
    #MA 30 CLOSE
    df['MA30'] = df['Close'].rolling(30).mean()
    #RSI 6 - Momentum
    df['RSI6'] = talib.RSI(df['Close'], 6)
    #RSI 12
    df['RSI12'] = talib.RSI(df['Close'], 12) ##
    #RSI 24
    df['RSI24'] = talib.RSI(df['Close'], 24) ##
    #Bollinger Bands - Volatility
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], matype=talib.MA_Type.T3)        
    #Money flow index - momentum
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    #On Balance Volume - Volume
    df['OnBalanceVolume'] = talib.OBV(df['Close'], df['Volume'])
    
    dropna = False
    if dropna: df = df.dropna()
    else: df = df.fillna(0) #.fillna(0) #(method='ffill')
    
    scaler = None
    df_norm = None
        
    y = df['Close_Tomorrow']
    X = df.drop(["Close_Tomorrow"], axis=1)
    X_train, y_train  =  X[:testStartDate], y[:testStartDate]     #train_test_split(X, y, test_size=0.2, shuffle=False)
    X_test, y_test =        X[testStartDate:testEndDate], y[testStartDate:testEndDate]
    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        scaler_X_train = MinMaxScaler()
        scaler_y_train = MinMaxScaler()
        
        X_train_norm=pd.DataFrame(scaler_X_train.fit_transform(X_train),columns=X_train.columns, index=X_train.index)

        Y_train_norm= scaler_y_train.fit_transform(np.array(y_train).reshape(-1, 1))
        
        X_test_norm = pd.DataFrame(scaler_X_train.transform(X_test),columns=X_test.columns, index=X_test.index)
        y_test_norm = scaler_y_train.transform(np.array(y_test).reshape(-1, 1))
        
        return (df, X_train, X_test, y_train, y_test, 
               X_train_norm, Y_train_norm, X_test_norm, y_test_norm, 
               scaler_X_train, scaler_y_train)

    else: 
        return df, X_train, X_test, y_train, y_test 

def combinePrediction(y_test, df, pred, col):
    pred = pred.reshape(pred.shape[0], -1)
#predicted
    p = []
    for i in pred:
        p.append(i[0])
    
    import pandas as pd
    result = pd.DataFrame()
    result['Pred_' + col] = pd.Series(p)
    result.index = y_test.index
#result['Original'] = pd.Series(o)
    combined = pd.concat([df, result], ignore_index=False, axis=1)
#combined = pd.concat([combined, y], ignore_index=False, axis= 1)
    df = combined
    return df, result

def fitAndPredict(modelName,df, X_train, X_test, y_train, y_test, 
        X_train_norm, Y_train_norm, X_test_norm, y_test_norm, 
        scaler_X_train, scaler_y_train):    
    
    import statistics as s
    
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    model = None
    if modelName == 'LinReg':
        from sklearn.linear_model import LinearRegression
        model = GridSearchCV(estimator=LinearRegression(),cv=tscv,param_grid={},scoring = 'neg_mean_squared_error')
    if modelName == 'SVRL':
        from sklearn.svm import SVR
        param_grid = {'C' : [ 0.001, 0.01, 0.1, 1 ]}
        model = GridSearchCV(estimator=SVR(kernel = 'linear'), 
                             cv=tscv,param_grid=param_grid, scoring = 'neg_mean_squared_error') 
    if modelName == 'SVRP':
        from sklearn.svm import SVR
        param_grid = {'C' : [ 0.001, 0.01, 0.1, 1 ]}
        model = GridSearchCV(estimator=SVR(kernel = 'poly'), 
                             cv=tscv,param_grid=param_grid, scoring = 'neg_mean_squared_error')
    if modelName == 'SVRRBF':
        from sklearn.svm import SVR
        param_grid = {'C' : [ 0.001, 0.01, 0.1, 1 ]}
        model = GridSearchCV(estimator=SVR(kernel = 'rbf'), 
                             cv=tscv,param_grid=param_grid, scoring = 'neg_mean_squared_error')
    if modelName == 'MLP':
        from sklearn.neural_network import MLPRegressor
        param_grid = { 'max_iter': [250,500,1000], 
                      'learning_rate_init': [0.01,0.001],
                       'hidden_layer_sizes': [(10,10,10),(25,25,25),(50,50,50),
                                              (10,10,10,10,10),(25,25,25,25,25),(50,50,50,50,50) ]
                     }
        model = GridSearchCV(estimator=MLPRegressor(early_stopping = True), 
                             cv=tscv,param_grid=param_grid, scoring = 'neg_mean_squared_error')
    if modelName == 'XGBoost':
        from xgboost import XGBRegressor
        #model = XGBRegressor()
        
        param_grid = { 'booster': ['gbtree','gblinear','dart'], 
                       'max_delta_step': [0,1,5],
                       'lambda':[1,3,5,10,50,100]
                     }
        model = GridSearchCV(estimator= XGBRegressor(),
                             cv=tscv,param_grid=param_grid,scoring = 'neg_mean_squared_error')
    if modelName == 'RandForest':
        from sklearn.ensemble import RandomForestRegressor
        #model = RandomForestRegressor() #max_depth=2, random_state=0)
        
        param_grid = {'criterion':['squared_error','poisson'],
                       'n_estimators': [150],
                       'max_leaf_nodes':[5,10,35,None],
                       'min_samples_leaf':[1,3,5]}
        model = GridSearchCV(estimator= RandomForestRegressor() ,
                             cv=tscv,param_grid=param_grid,scoring = 'neg_mean_squared_error')

        
    if modelName == 'GradBoost':
        from sklearn.ensemble import GradientBoostingRegressor
        param_grid = {
            'criterion' : ['friedman_mse'], 
            'n_estimators' : [150],
            'learning_rate' : [0.001, 0.01, 0.1],
             'max_depth' : [3, 5, 10], 
             'max_leaf_nodes' : [5, 10, 35, None], 
             'min_samples_leaf' : [1, 3, 5]
            }
        model = GridSearchCV(estimator= GradientBoostingRegressor(),
                             cv=tscv,param_grid=param_grid,scoring = 'neg_mean_squared_error')

    error_norm = []
    error_unorm = []
    model.fit(X_train_norm.to_numpy() , Y_train_norm.ravel())
    
    report = {'Best_Params'         : str(model.best_params_),
             'Train - Best Score:': str(model.best_score_),
             'Test - Score:': str(model.score(X_train_norm.to_numpy(), Y_train_norm.ravel()))
     }
    pred = model.predict(X_test_norm.to_numpy())
    mse_norm, rmse_norm = calculateError(y_test_norm,pred)
    pred_unorm = scaler_y_train.inverse_transform(np.array(pred).reshape(-1,1))
    df, result = combinePrediction(y_test, df, pred_unorm, modelName)
    mse_unorm, rmse_unorm = calculateError(y_test, result['Pred_'+modelName])
    report_2 = {'test_error_norm_mse': mse_norm, 'test_error_unorm_mse': mse_unorm}
    report.update(report_2)
    return df, result, report
    

def movement(df, col):
    movement = []
    movement_pred = []
    for i, row  in df.iterrows():
        if row['Close_Tomorrow'] > row['Close']:    movement.append(1)
        elif row['Close_Tomorrow'] < row['Close']:  movement.append(-1)
        elif row['Close_Tomorrow'] == row['Close']: movement.append(0)
        
        if row['Pred_'+col] > row['Close']:    movement_pred.append(1)
        elif row['Pred_'+col] < row['Close']:  movement_pred.append(-1)
        elif row['Pred_'+col] == row['Close']: movement_pred.append(0)
        else: movement_pred.append(np.NaN) 

    df['Movement_Real'] = movement
    df['Movement_Pred_'+col] = movement_pred
    
    hit = []
    for i, row in df.iterrows():
        if pd.isna(row['Movement_Pred_'+col]) == False:
            if row['Movement_Real'] == row['Movement_Pred_'+col]: hit.append(1)
            else: hit.append(0)
        else: hit.append(np.NaN)
        
    df['Hit_Pred_'+col] = hit
 
    return df

def calculateError(y_true, y_pred): 
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    return mse, rmse


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    print('------------- Data Info --------------')
    df = loadData(filename)
    print(df.info())    
    df.describe().to_excel("./Output/Describe_"+year+".xlsx")
    fig = df[:testEndDate].plot(y = 'Close', figsize = (15,5)).get_figure()
    fig.savefig('./Output/FullGraph_'+year)
    
    fig = df[testStartDate:testEndDate].plot(y = 'Close', figsize = (15,5)).get_figure()
    fig.savefig('./Output/FullGraph_Test'+testStartDate+'_'+testEndDate)
    print('------------- ----------------- --------------')

    dataFrameResult = pd.DataFrame()    
    dataFrameResult = runModel('LinReg', filename, dataFrameResult)
    #dataFrameResult = runModel('SVRL', filename, dataFrameResult)
    #dataFrameResult = runModel('SVRP', filename, dataFrameResult)
    #dataFrameResult = runModel('SVRRBF', filename, dataFrameResult)
    #dataFrameResult = runModel('MLP', filename, dataFrameResult)
    #dataFrameResult = runModel('RandForest', filename, dataFrameResult)
    #dataFrameResult = runModel('GradBoost', filename, dataFrameResult)
    #dataFrameResult = runModel('XGBoost', filename, dataFrameResult)
    
    import backtrader as bt
    import Trader as trade
    data = bt.feeds.YahooFinanceCSVData(
            dataname=filename, 
            fromdate=pd.to_datetime(testStartDate),    #cst.period_03_train['startTest'],
            todate=pd.to_datetime(testEndDate))      #cst.period_03_train['endTest'])
    
    report = '\n*********** BASELINES ***********'
    print('\n----------\nBuyHold\n')
    from Trader import BuyAndHold
    report = trade.createRunCerebro(data, BuyAndHold, 'Buy Hold', 'Buy_Hold'+testStartDate+'_'+testEndDate)
    report.update({'model': 'Buy&Hold'})
    dataFrameResult = dataFrameResult.append(report, ignore_index=True)
    
    print('Mov Avg\n')
    from Trader import MovAVGStrategy
    try:
        report = trade.createRunCerebro(data, MovAVGStrategy, 'MA Crossover', 'MA_'+testStartDate+'_'+testEndDate)
        report.update({'model': 'Mov Avg'})
        dataFrameResult = dataFrameResult.append(report, ignore_index=True)
        print(report)
    except Exception as e:
        report = '/*/*/*/*/* MA Exception /*/*/*/*/*'+str(e)
        print('MA Exception:',e)
    
    dataFrameResult = dataFrameResult.sort_values(by = 'Value', ascending=False)    
    print(dataFrameResult)    
    dataFrameResult.to_excel('./Output/FinalResult_'+year+'.xlsx')