# coding: utf-8

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    dataSetSize = dataset.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDisIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    for i in classCount.iteritems():
        print(i)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
