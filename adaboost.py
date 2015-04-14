#coding=utf-8
#!/usr/bin/python
from numpy import *
import os,sys
import datetime
import time
def loadSimpleData():
    dataMat = matrix([[1. , 2.1],
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArry = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArry[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArry[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArry

######D��Ȩ������
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0#���������п���ֵ�ϱ���
    bestStump = {}#���ڴ洢�������������Ϣ
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):#������������
        rangeMin = dataMatrix[0:m,i].min()
        rangeMax = dataMatrix[0:m,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)#�õ���ֵ
                #���ݷ�ֵ����
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr#��ͬ������Ȩ���ǲ�һ����
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i 
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m =shape(dataArr)[0]
    D = mat(ones((m,1))/m)                          #��ʼ������������Ȩֵһ��
    aggClassEst = mat(zeros((m,1)))                 #ÿ�����ݵ�Ĺ���ֵ
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #����alpha��max(error,1e-16)��֤û�д����ʱ�򲻳��ֳ������
        #alpha��ʾ���������������Ȩ�أ�������Խ�ͷ�����Ȩ��Խ��
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: 
            break
    return weakClassArr

#dataToClass ��ʾҪ����ĵ��㼯
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)            #do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
#        print (aggClassEst)
    return sign(aggClassEst)

def SplitStr(str,labels,feavect):
    strvalue = str.split()
    feaval = []
    labels.append(int(strvalue[0]))
    for i in range(len(strvalue) - 1):
        temp = strvalue[i+1].split(':')
        feaval.append(float(temp[1]))
    feavect.append(feaval)
    return strvalue


def LoadTestData(name):
    try:
        FileObject = open(name,'r')
    except IOError:
        print ("Flie open error")
        exit()
    ListAllLines = FileObject.readlines()
    FileObject.close()
    labels = []
    feavect = []
    for EachLine in ListAllLines:
        SplitStr(EachLine,labels,feavect)
    return feavect,labels
def main():
    print ("PythonApplication")
    labels = []
    feavect = []
    starttime = time.clock()
    feavect,labels = LoadTestData('E:/Actual/train12.txt')
    LoadTestDataTime = time.clock()
    print("Load TrainText cost time: "+str(LoadTestDataTime-starttime)+" s")
    D = mat(ones((1152,1))/1152)
    BoostTrainStartTime = time.clock()
    classifierArr = adaBoostTrainDS(feavect,labels,30)
    BoostTrainEndTime = time.clock()
    print("Adaboost Train cost time: "+str(BoostTrainEndTime-BoostTrainStartTime)+" s")
    testfeavect,testlables = LoadTestData('E:/Actual/test12.txt')
    LoadTestDataTime = time.clock()
    print("Load TestData cost time: "+str(LoadTestDataTime-BoostTrainEndTime)+" s")
    count = 0
    for i in range(len(testfeavect)):
        t = adaClassify(feavect[i],classifierArr)
        if testlables[i] == t.min():
            count = count + 1
#        print (t) 
    print("Adaboost Classify cost time: "+str(time.clock()-LoadTestDataTime)+" s")
    print("Total time: "+ str(time.clock() - starttime) + "s")
    print ("Accuray: "+str(count/len(testfeavect)*100) + "%")

    


if __name__ == '__main__':
    main()