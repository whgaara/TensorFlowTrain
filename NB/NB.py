# coding: utf-8

from numpy import *


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 去重，选取所有词汇
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 给出一个所有词汇的状态图，如果输入的序列在词汇表中，输出的词汇表状态中对应位置为1，就是词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 求0/1条件下，各个词的条件概率
    # 拉普拉斯转移并取log，为了将连乘转成log的求和
    p1Vect = log(p1Num + 1 / (p1Denom + 6))
    p0Vect = log(p0Num + 1 / (p0Denom + 6))
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对连乘取log
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p2 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p2:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, classifyNB(thisDoc, p0v, p1v, pAb))


testingNB()
