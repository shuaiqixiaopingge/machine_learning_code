from numpy import *
import operator
import knn
import os
import matplotlib
import matplotlib.pyplot as plt

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, (32*i + j)] = int(lineStr[j])
    return returnVect

def handwritingClassTest(k):
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorcount = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = knn.classify0(vectorUnderTest, trainingMat, hwLabels,k)
        #print('the classifier came back with: %d, the real number is %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorcount += 1.0
    #print('\nthe total number of errors is: %s' % int(errorcount))
    #print('\nthe total error rate is: %s' % float(errorcount/mTest))
    return float(errorcount/mTest)

def kError():
    errorList = []
    klist = []
    for k in range(2, 20):
        klist.append(k)
        error = handwritingClassTest(k)
        print(error)
        errorList.append(error)
    #fig = plt.figure()
    plt.plot(klist, errorList)
    plt.savefig('1.jpg')
    #plt.show()

kError()
