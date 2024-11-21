import numpy as np
import pandas as pd
import math


def squaredError(gx, yvalidate):
	se = pow((yvalidate - gx),2)
	return se

def weights(x,y):
	x_xT_inv = np.linalg.inv(np.dot(x.T, x))
	xT_y = np.dot(x.T, y)
	theta = np.dot(x_xT_inv, xT_y)
	return theta

def parseData():
    dfX = pd.read_csv('insurance.csv', sep=',', usecols=['age','sex','bmi','children','smoker','region', 'charges'])
    dfX['sex'].replace(['male', 'female'],[0, 1], inplace=True)
    dfX['smoker'].replace(['yes', 'no'],[0, 1], inplace=True)
    dfX['region'].replace(['southwest', 'southeast', 'northwest', 'northeast'],[0, 1, 2, 3], inplace=True)
    return dfX.values

def rmse(gx, yvalidate):
    se = 0
    for i in range(np.size(gx, axis=0)):
        se = se + math.pow(yvalidate[i] - gx[i],2)
    mse = se/np.size(gx, axis=0)
    RMSE = math.sqrt(mse)
    return RMSE
	

def sFolds(S):
	data = parseData()
	stopper = math.ceil(len(data) / S)	
	RMSE = []
	
	for runs in range(20):	
		np.random.shuffle(data)
		folds = []
		rmsearr = []
		temp = data[:]
		for K in range(S):
			folds.append(temp[0:stopper])
			temp = np.delete(temp,slice(0,stopper), axis=0)
	
		for i in range(S): 
			temp = folds[:]
			testing_data = np.asarray(temp[i])
			del temp[i]
			training_data = np.asarray(temp)		
			training_data = np.concatenate([training_data[f] for f in range(len(training_data))],axis=0)
			""" print(testing_data)
			print(training_data) """
			
			yvalidate = testing_data[:,-1]
			xvalidate = testing_data[:,:-1]
			ytrain = training_data[:,-1]
			xtrain = training_data[:,:-1]
			#print(yvalidate.size, xvalidate.size, ytrain.size, xtrain.size)
			""" print(yvalidate)
			print(xvalidate)
			print(ytrain)
			print(xtrain) """
			
			xtrain = np.insert(xtrain, 0, 1, axis=1)
			xvalidate = np.insert(xvalidate, 0,1, axis=1)

			theta = weights(xtrain,ytrain)
			""" print(xvalidate)
			print(yvalidate)
			print(xtrain)
			print(ytrain)
			print(theta) """

			gx = np.dot(xvalidate,theta)
			rmseval = rmse(gx, yvalidate)
			rmsearr.append(rmseval)
		RMSE.append(rmsearr)
		
	RMSE = np.asarray(RMSE)
	#print(RMSE)
	print(str.format("When S = {}, Mean RMSE = {} and the Standard Deviation is = {}", S, RMSE.mean(), RMSE.std()))

def main():
    sFolds(4)
    sFolds(11)
    sFolds(22)
    df = pd.read_csv('insurance.csv', sep=',', usecols=['age','sex','bmi','children','smoker','region', 'charges'])
    N = len(df.index)
    sFolds(N)

if __name__ == "__main__" :
    main()