import math, random
import numpy as np
np.random.seed(0)
learn = 0.1

#first 3 functions are the same as bayes.py. Described in detail there
def standardize(data):
    stand = (data)
    m = data.mean(axis=0)
    std = data.std(axis=0,ddof=1)

    stand = (stand-m) / std
    return stand

def standardizeWithTrain(data, tr_mean, tr_std):
	stand = (data)
	m = tr_mean
	std = tr_std
	stand = (stand-m) / std
	return stand

def parseAndClassify(data):
	# get data from file
    full_data = np.genfromtxt(data, delimiter=',')
	# shuffle
    np.random.shuffle(full_data)

	# make a copy of the full data to make the separation of training (2/3) and testng (1/3) 
    temp = full_data[:]
    training_data = temp[0:math.ceil(np.size(temp,axis=0)*(2/3))]
    temp = np.delete(temp,slice(0,np.size(training_data,axis=0)), axis=0)
    testing_data = temp	
    training_stand = standardize(training_data[:,:-1])
    testing_data_label = testing_data[:, -1]
    training_data_label = training_data[:,-1]
    testing_data = standardizeWithTrain(testing_data[:,:-1], training_data[:,:-1].mean(axis=0), training_data[:,:-1].std(axis=0,ddof=1))
    return training_stand, testing_data, testing_data_label, training_data_label

###Statistics!
def precision(tp,fp):
	p = tp/(tp+fp)
	print(str.format("Precision: {}", p))
	return p

def recall(tp,fn):
	r = tp/(tp+fn)
	print(str.format("Recall: {}", r))
	return r

def fMeasure(pr,re):
	f = 2*pr*re / (pr + re) 
	print(str.format("f-Measure: {}", f))
	return f

def accuracy(testing_data,true_p, true_n):
	acc = (1/np.size(testing_data, axis=0)) * (true_p+true_n)
	print(str.format("accuracy: {}", acc))
	return acc


#Classifying with the predicition and the true label
def binClassification(testing_data,true_class, pred_class):
	true_p = 0
	true_n = 0
	false_p = 0
	false_n = 0
	
	for i in range(len(pred_class)):
		if true_class[i] == pred_class[i] and true_class[i] == 1:
			true_p +=1
		elif true_class[i] == pred_class[i] and true_class[i] == 0:
			true_n += 1
		elif true_class[i] != pred_class[i] and true_class[i] == 1:
			false_n +=1
		elif true_class[i] != pred_class[i] and true_class[i] == 0:
			false_p +=1
	return true_p, true_n, false_p, false_n

# P (class = 1 given x)
# for negative class P (class = 0 given x) : 1-sigmoid
def sigmoid(theta, data):
    xTheta = np.dot(data,theta )
    sigmoid = 1/(1+ np.exp(-xTheta))
    if isinstance(sigmoid, np.float_) == True:
        if sigmoid <= 0.0000000001:
            return 0.0000000001
    return sigmoid

#computing theta
def weights(x_data, y_labels, theta):
    gx = sigmoid(theta, x_data)
    new_theta = theta + (learn/np.size(x_data,axis=0)) * (np.dot(x_data.T,(np.asmatrix(y_labels).T - gx)))
    theta = new_theta
    return theta

#computes the log likelihood. Also checking for domain errors here
def getLogLike(testing_data, testing_data_label, theta):
    log_like = 0
    for row in range(np.size(testing_data, axis=0)):
        if np.asmatrix(sigmoid(theta,testing_data[row])).shape == (1,1) and np.mean(1-sigmoid(theta,testing_data[row])) <=0:
            log_like +=  0.00000001
        elif np.asmatrix(sigmoid(theta,testing_data[row])).shape == (1,1) and np.mean(sigmoid(theta,testing_data[row])) <=0:
            log_like +=  0.00000001
        else:
            log_like += testing_data_label[row] * math.log(sigmoid(theta,testing_data[row])) + (1-testing_data_label[row]) *  math.log((1-sigmoid(theta,testing_data[row])))
    return log_like

#classifies the trained thetas with the testing data
def classify(testing_data, theta):
    predicted_data= {}
    classified = sigmoid(theta, testing_data)
    classified = classified * 100
    for i in range(np.size(classified)):
        
        if classified[i,0] >= 50.0:
           predicted_data[i] = 1
        else:
            predicted_data[i] = 0
    return predicted_data

#main functuion. Computes everything necessary to get the statistics
def logReg(data):
    training_data, testing_data, testing_data_label, training_data_label = parseAndClassify(data)

    #add leading 1 bias
    training_data = np.insert(training_data, 0,1, axis=1)
    testing_data = np.insert(testing_data,0,1, axis=1)

    #randomize theta
    theta = [[random.uniform(-1,1)] for col in range(np.size(testing_data, axis=1))]
    log_like = getLogLike(testing_data, testing_data_label, theta)
    change = 100
    count = 0 

    #termination condition. Train thetas, change the log likelihood to see if it should terminate. count the runs
    while abs(change) > 1:
        theta = weights(training_data, training_data_label,theta)
        temp_log_like = getLogLike(testing_data,testing_data_label,theta)
        change = log_like-temp_log_like
        log_like = temp_log_like
        count += 1
    print(count)

    #the result of the prediction
    predicted_class = classify(testing_data,theta)

    #measures for statistics
    true_p, true_n, false_p, false_n = binClassification(testing_data, testing_data_label, predicted_class)
    acc = accuracy(testing_data, true_p, true_n)
    pr = precision(true_p, false_p)
    re = recall(true_p, false_n)
    f_m = fMeasure(pr, re)


logReg('spambase.data')