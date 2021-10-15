import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def main():
    tr = np.load('../data/CWRU/train.npz')
    te = np.load('../data/CWRU/test.npz')
    x_train, y_train = tr['X_train'], tr['Y_train']
    x_test,  y_test  = te['X_test'],  te['Y_test']

    clf = LogisticRegression(max_iter=10000)
    clf.fit(x_train, y_train)
    y = clf.predict(x_test)

    acc = accuracy_score(y_test, y)
    print(acc)


if __name__ == '__main__':
    main()