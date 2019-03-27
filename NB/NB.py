# coding: utf-8

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0]
    return postingList, classVec


# 去重，选取所有词汇
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 给出一个所有词汇的状态图，如果输入的序列在词汇表中，输出的词汇表状态中对应位置为1
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!'%word)
    return returnVec


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
# print(myVocabList)
print(setOfWords2Vec(myVocabList, listOPosts[0]))











