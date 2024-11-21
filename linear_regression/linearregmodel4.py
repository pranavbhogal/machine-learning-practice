import numpy as np
import pandas as pd
import math 

np.random.seed(0)

def standardize(matrix):
   stand = matrix
   mean = matrix.mean(axis=(0), keepdims=False)
   sd = matrix.std(axis=(0), keepdims=False, ddof = 1)
   stand = (stand-mean) / sd
   return stand, matrix.mean(axis=(0), keepdims=False), matrix.std(axis=(0), keepdims=False, ddof=1)

#compute theta = (xT * x)^-1 (xT * y)
def weights(x,y):
	x_xT_inv = np.linalg.pinv(np.dot(x.T, x))
	xT_y = np.dot(x.T, y)
	theta = np.dot(x_xT_inv, xT_y)
	
	return theta

def smape(gx, yvalidate):
    sum = 0
    for i in range(np.size(gx, axis=0)):
         sum = sum + abs(yvalidate[i] - gx[i])/(abs(yvalidate[i])+abs(gx[i]))
    smapeval = sum/np.size(gx, axis=0)
    return smapeval


def rmse(gx, yvalidate):
	#se = math.pow((sum(yvalidate) - sum(gx)),2)
	#mse = se/np.size(gx, axis=0)
    se = 0
    for i in range(np.size(gx, axis=0)):
        se = se + math.pow(yvalidate[i] - gx[i],2)
    mse = se/np.size(gx, axis=0)
    rmseval = math.sqrt(mse)
    return rmseval

def closedFormLinReg(xtrain, ytrain, xvalidate, yvalidate):
    theta = weights(xtrain,ytrain)
    #print(theta)
    gx = np.dot(xvalidate,theta)
    #print(np.size(gx, axis=0))
    #print(np.size(yvalidate, axis=0))
    rmseval = rmse(gx, yvalidate)
    print("Root Mean Squared Error: ",rmseval)
    smapeval = smape(gx, yvalidate)
    print("Symmetric mean absolute percentage error: ",smapeval)

def main():
    dfX = pd.read_csv('insurance.csv', sep=',', usecols=['age','sex','bmi','children','smoker','region'])
    dfX['sex'].replace(['male', 'female'],[0, 1], inplace=True)
    dfX['smoker'].replace(['yes', 'no'],[0, 1], inplace=True)
    #dfX['region'].replace(['southwest', 'southeast', 'northwest', 'northeast'],[0, 1, 2, 3], inplace=True)
    dfbinregion = pd.get_dummies(dfX["region"])
    dfX = pd.concat((dfbinregion, dfX), axis=1)
    dfX = dfX.drop(["region"], axis=1)
    dfX.insert(loc=0, column='bias', value=1)
    #print(dfX)
    #print(dfX.values)

    dfY = pd.read_csv('insurance.csv', sep=',', usecols=['charges'])
    #print(dfY.values)

    #dfX, dfXMean, dfXSd = standardize(dfX)
    #dfX = dfX.sample(frac = 1)
    xtrain, xvalidate = np.split(dfX.sample(frac=1, random_state=42), [int(.67*len(dfX))])
    ytrain, yvalidate = np.split(dfY.sample(frac=1, random_state=42), [int(.67*len(dfY))])
    print('validation metrics')
    closedFormLinReg(xtrain.values, ytrain.values, xvalidate.values, yvalidate.values)
    print('training metrics')
    closedFormLinReg(xtrain.values, ytrain.values, xtrain.values, ytrain.values)

if __name__ == "__main__" :
    main()