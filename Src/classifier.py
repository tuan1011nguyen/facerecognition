from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_classifier(X_train, y_train):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    clf = SVC(C=5.0, gamma=0.001)
    clf.fit(X_train, y_train_encoded)
    return clf, le

def predict(clf, le, X_test, y_test):
    y_test_encoded = le.transform(y_test)
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_predict)
    return accuracy, le.inverse_transform(y_predict)
