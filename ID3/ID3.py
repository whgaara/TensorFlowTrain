# coding: utf-8

from math import log
import operator

"""
数据样本：
编号         用腮呼吸         是否有脚蹼          属于鱼类
1              是               是                 是
2              是               否                 是
3              是               否                 否
4              否               是                 否
5              否               是                 否
"""


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 0, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['gill', 'filppers']
    return dataSet, labels


def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 提取了第一列特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 对i个特征值根据value进行数据集的划分
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 对照西瓜书，可以知道每个特征值信息量求解的公式如下
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            # 会告诉你根据哪个特征划分是最好的选择
            bestFeature = i
    return bestFeature


# 投票机，选出占比最大的那个类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(zip(tuple(classCount), tuple(classCount.values())),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果全是一样的结果就没有计算的必要，例如['no', 'no', 'no']，此时另一个分支还存在子节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果只有一个特征，那就直接返回特征最多的那个分类，到最后一层使用，例如['yes', 'no', 'no']，最后一次分类触发
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 建立决策树的根节点分类
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    # 进入下一层节点进行分类
    for value in uniqueVals:
        subLabels = labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


dataSet, labels = createDataSet()
# print(calShannonEnt(dataSet))
# print(chooseBestFeatureToSplit(dataSet))
# print(majorityCnt(['yes', 'no']))
print(createTree(dataSet, labels))
