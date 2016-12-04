from sklearn.linear_model import SGDClassifier
import numpy as np

import csv
import chess

import datetime

from features import ALL_FEATURES

def datasetBatches(filename, batchSize=10):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) == batchSize:
                yield rows
                rows = []
        yield rows

def extractFeatureVector(board):
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

def resultClass(resultString):
    if resultString == "1-0":
        return 1
    else:
        return -1

def train(filename):
    regressor = SGDClassifier(fit_intercept=False, loss="log")
    for batch in datasetBatches(filename, batchSize=10000):
        trainingExamples = []
        labels = []
        for epd, result, _ in batch:
            if result == "0-0":
                print "Warning: draw match"
                continue # Skip draws
            board = chess.Board()
            board.set_epd(epd)
            featureVector = extractFeatureVector(board)
            trainingExamples.append(featureVector)
            label = resultClass(result)
            labels.append(label)
        X = np.vstack(trainingExamples)
        y = np.array(labels)
        regressor.partial_fit(X, y, classes=[-1, 1])
    return regressor.coef_

def test(filename, weights):
    correctCount = 0
    totalCount = 0
    for batch in datasetBatches(filename, batchSize=10000):
        testExamples = []
        labels = []
        for epd, result, _ in batch:
            if result == "0-0":
                print "Warning: draw match"
                continue # Skip draws
            board = chess.Board()
            board.set_epd(epd)
            featureVector = extractFeatureVector(board)
            testExamples.append(featureVector)
            label = resultClass(result)
            labels.append(label)
        X = np.vstack(testExamples)
        y = np.array(labels)
        guesses = np.dot(X, weights.T).flatten()
        guesses = ((guesses > 0) * 2) - 1
        matches = y + guesses
        length = y.shape[0]
        correctCount += np.count_nonzero(matches)
        totalCount += length

    print "Correct: ", correctCount
    print "Total: ", totalCount
    return correctCount / float(totalCount)

def main():
    weights = train("./train/sample.csv")
    print weights
    now = datetime.datetime.now()
    np.save("./train/{} pretrained weights".format(now), weights)

    # weights = np.load("./train/2016-12-03 17:45:55.697714 pretrained weights.npy")\
    # Compute training set error
    test("./train/sample.csv", weights)


if __name__ == "__main__":
    main()
