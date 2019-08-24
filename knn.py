import numpy as np
import operator

def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
#k近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #计算欧式距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffmat = diffMat**2
    sqDistances = sqDiffmat.sum(axis = 1)
    distances = sqDistances**0.5
    #argsort函数将按照既定顺序（默认为升序）将数组元素排列并提取其索引值
    sortDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortDistIndices[i]]
        #dict.get(key,default=None)返回值为key在dict中的value，如果没有key则返回default值
        #计算各label出现的次数，返回一dict
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #将dict转换为tuple，并按照label出现的次数降序排列
    sortedClasscount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasscount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    #需要先生成numpy的数列
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        #删除字符开头结尾处的空格和换行符，readlines读到的line的结尾处有换行符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        else:
            classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
def autonorm(dataSet):
    #np.min(0)将会按照列求解最小值，此例中将返回一行向量
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    m = dataSet.shape[0]
    normDataSet = np.zeros(np.shape(dataSet))
    #tile(A,reps)将A值重复reps次构成的numpy矩阵，其中，A，reps可为矩阵。
    normDataSet = dataSet - np.tile(minvals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, minvals, ranges
#利用的数据集中10%数据进行测试
def datingClassTesting():
    hoRatio = 0.10
    datingMat, ClassLabels = file2matrix('datingTestSet.txt')
    errors = 0.0
    normMat, minVals, ranges = autonorm(datingMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], ClassLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, ClassLabels[i]))
        if (classifierResult != ClassLabels[i]):
            errors += 1.0
    print('the total error rate is %f' % (errors/float(numTestVecs)))

def classifyPerson():
    resultist = ['didntLike', 'smallDoses', 'LargeDoses']
    ffMiles = float(input('frequent flight miles earned per year: '))
    percentVideo = float(input('percentage of time playing video game:'))
    litterIcecream = float(input('liters of ice-cream persumed per year: '))
    inArray = np.array([ffMiles, percentVideo, litterIcecream])
    dataSet, dataLabel = file2matrix('datingTestSet2.txt')
    normdataSet, minVals, ranges = autonorm(dataSet)
    labelNum = classify0((inArray - minVals)/ranges, normdataSet, dataLabel, 3)
    return resultist[labelNum - 1]

#datingClassTesting()
