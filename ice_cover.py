# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:39:19 2020

@author: pulki
"""
import csv
import math
import numpy as np
import random

def get_dataset():
    dictList = []
    with open('./MendotaLake.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dictList.append(row)
    
    for i in range(3, len(dictList)):
        if(dictList[i]['WINTER'] != '"'):
            dictList[i]['WINTER'] = int(dictList[i]['WINTER'][0:4])
        if(dictList[i]['DAYS'][0:1] != '-'):
            dictList[i]['DAYS'] = int(dictList[i]['DAYS'])
    
    ret= [[0]*2 for i in range(len(dictList))]
    j = 0
    for i in range(3, len(dictList)):
        if(dictList[i]['WINTER'] != '"'):
            ret[j][0] = dictList[i]['WINTER']
            j = j + 1
    j = 0
    for i in range(3, len(dictList)):
        if(isinstance(dictList[i]['DAYS'], int)):
            ret[j][1] = dictList[i]['DAYS']
            j = j + 1
    retu = []
    for i in range(len(ret)):
        if(ret[i][0] != 0):
            retu.append(ret[i])
            
    return retu

def print_stats(dataset):
    length = len(dataset)
    print(length)
    tot = 0
    for i in range(len(dataset)):
        tot = tot + dataset[i][1]
    mean = tot/length
    print('{0:.2f}'.format(mean))
    dev = 0
    for i in range(len(dataset)):
        dev = dev + (dataset[i][1] - mean) * (dataset[i][1] - mean)
    std = dev/(length - 1)
    ret = math.sqrt(std)
    print('{0:.2f}'.format(ret))
    
def regression(beta_0, beta_1):
    data = get_dataset()
    length = len(data)
    sum = 0
    for i in range(len(data)):
        sum = sum + (beta_0 + beta_1 * data[i][0] - data[i][1]) * (beta_0 + beta_1 * data[i][0] - data[i][1])
    mse = sum/length
    return mse
    
def gradient_descent(beta_0, beta_1):
    data = get_dataset()
    length = len(data)
    sum = 0
    sum2 = 0
    for i in range(len(data)):
        sum = sum + (beta_0 + beta_1 * data[i][0] - data[i][1])
        sum2 = sum2 + (beta_0 + beta_1 * data[i][0] - data[i][1]) * data[i][0]
    x = sum * 2/length
    y = sum2 * 2/length
    tup = (x,y)
    return tup

def iterate_gradient(T, eta):
    xprev = 0
    yprev = 0
    for i in range(T):
        tup = gradient_descent(xprev, yprev)
        x = xprev -eta * tup[0]
        y = yprev -eta * tup[1]
        mse = regression(x,y)
        print(i + 1, '{0:.2f}'.format(x) , '{0:.2f}'.format(y), '{0:.2f}'.format(mse))
        xprev = x
        yprev = y
        
def compute_betas():
    data = get_dataset()
    length = len(data)
    xsum = 0
    ysum = 0
    for i in range(len(data)):
        xsum = xsum + data[i][0]
        ysum = ysum + data[i][1]
    xmean = xsum/length
    ymean = ysum/length
    ytop = 0
    ybot = 0
    for i in range(len(data)):
        ytop = ytop + (data[i][0] - xmean) * (data[i][1] - ymean)
        ybot = ybot + (data[i][0] - xmean) * (data[i][0] - xmean)
    beta_1 = ytop/ybot
    beta_0 = ymean - beta_1 * xmean
    mse = regression(beta_0, beta_1)
    tup = (beta_0, beta_1, mse)
    return tup

def predict(year):
    beta_0, beta_1, mse = compute_betas()
    days = beta_0 + beta_1 * year
    num = '{0:.2f}'.format(days)
    return float(num)

def gradient_descent_helper(beta_0, beta_1,data):
    #data = get_dataset()
    length = len(data)
    sum = 0
    sum2 = 0
    for i in range(len(data)):
        sum = sum + (beta_0 + beta_1 * data[i][0] - data[i][1])
        sum2 = sum2 + (beta_0 + beta_1 * data[i][0] - data[i][1]) * data[i][0]
    x = sum * 2/length
    y = sum2 * 2/length
    tup = (x,y)
    return tup

def regression_helper(beta_0, beta_1, data):
    #data = get_dataset()
    length = len(data)
    sum = 0
    for i in range(len(data)):
        sum = sum + (beta_0 + beta_1 * data[i][0] - data[i][1]) * (beta_0 + beta_1 * data[i][0] - data[i][1])
    mse = sum/length
    return mse

def iterate_normalized(T, eta):
    data = get_dataset()
    length = len(data)
    xsum = 0
    for i in range(len(data)):
        xsum = xsum + data[i][0]
    xmean = xsum/length
    
    dev = 0
    for i in range(len(data)):
        dev = dev + (data[i][0] - xmean) * (data[i][0] - xmean)
    std = dev/(length - 1)
    xdv = math.sqrt(std)
    
    for i in range(len(data)):
        data[i][0] = (data[i][0] - xmean) / xdv
       
    xprev = 0
    yprev = 0
    for i in range(T):
        tup = gradient_descent_helper(xprev, yprev,data)
        x = xprev -eta * tup[0]
        y = yprev -eta * tup[1]
        mse = regression_helper(x,y,data)
        print(i + 1, '{0:.2f}'.format(x) , '{0:.2f}'.format(y), '{0:.2f}'.format(mse))
        xprev = x
        yprev = y
        
def gradient_descent_random(beta_0, beta_1, data):
    #data = get_dataset()
    length = len(data)
    rand = random.randint(0, length - 1)
    #print(rand)
    x = 2 * (beta_0 + beta_1 * data[rand][0] - data[rand][1])
    y = 2 * (beta_0 + beta_1 * data[rand][0] - data[rand][1]) * data[rand][0]
    tup = (x,y)
    return tup

        
def sgd(T, eta):
    data = get_dataset()
    length = len(data)
    xsum = 0
    for i in range(len(data)):
        xsum = xsum + data[i][0]
    xmean = xsum/length
    
    dev = 0
    for i in range(len(data)):
        dev = dev + (data[i][0] - xmean) * (data[i][0] - xmean)
    std = dev/(length - 1)
    xdv = math.sqrt(std)
    
    for i in range(len(data)):
        data[i][0] = (data[i][0] - xmean) / xdv
        
    xprev = 0
    yprev = 0
    for i in range(T):
        tup = gradient_descent_random(xprev, yprev, data)
        x = xprev -eta * tup[0]
        y = yprev -eta * tup[1]
        mse = regression_helper(x, y, data)
        print(i + 1, '{0:.2f}'.format(x) , '{0:.2f}'.format(y), '{0:.2f}'.format(mse))
        xprev = x
        yprev = y


def main():
    get_dataset()
    
if __name__ == "__main__":
    main()






















        