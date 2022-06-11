# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
series = read_csv('daily-min-temperatures.csv', header=3285, index_col=0, parse_dates=True, squeeze=True)
# split dataset
X=series.values
j=90
predictions = list()
minus=list()
for i in range(275):
    train = X[i:(i + 90)]
    test =X[j:]
    # train autoregression
    window = 13
    model = AutoReg(train, lags=13)
    model_fit = model.fit()
    coef = model_fit.params
    #print(coef)
    # walk forward over time steps in test
    history = train[len(train) - 13:]
    #print(history)
    history = [history[i] for i in range(len(history))]
    #print(history)
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    #print(yhat)
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]
    obs = test[i]
    predictions.append(yhat)
    history.append(obs)
    minus.append(obs-yhat)
    print('predicted=%f, expected=%f ,Minus=%f' % (yhat, obs, (yhat-obs)))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.plot(minus,color='green')
pyplot.show()

