import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Vlad, you need to implement calculation of accuracy, recall and precision (i do not do it)
# This file contains configs for several tasks

# INput Data
input_file = "income_data.txt"

X = []
Y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 500

with open(input_file, "r") as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

print("---------[X After file reading]--------\n")
print(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
Y = X_encoded[:, -1].astype(int)

print("--------[X] ---- [Y]------\n")
print(X)
print(Y)

print("-----NORMALIZED-----\n")
scaller = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = scaller.fit_transform(X)
print(X)

# classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# classifier = OneVsOneClassifier(SVC(kernel="sigmoid"))
classifier = OneVsOneClassifier(SVC(kernel="poly", degree=8))
# classifier = OneVsOneClassifier(SVC(kernel="rbf"))
print("------CLASSIFIER One Vs One-----")
classifier.fit(X=X, y=Y)

X_train, X_test, y_train, y_test \
    = train_test_split(X, Y, test_size=0.2, random_state=5)

print("-----X_train, X_test, y_train, y_test -----")
print(X_train)
print(X_test)
print(y_train)
print(y_test)

print("-----NORMALIZED-----\n")
scaller = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_train = scaller.fit_transform(X_train)
print(X_train)

# It is neded for the 1,2,3,4 tasks
# classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# classifier = OneVsOneClassifier(SVC(kernel="sigmoid"))
# classifier = OneVsOneClassifier(SVC(kernel="poly", degree=8))
# classifier = OneVsOneClassifier(SVC(kernel="rbf"))

classifier.fit(X=X_train, y=y_train)
y_test_pred = classifier.predict(X_test)

print("-----Y_test_pred-----")
print(y_test_pred)

f1 = cross_val_score(classifier, X, Y, scoring="f1_weighted", cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
              'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = np.array([-1] * len(input_data))

print("LABEL ENCODERS")
print(input_data_encoded)
count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = item
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item]))
        count += 1

input_data_encoded = input_data_encoded.astype(int)
input_data_encoded = [input_data_encoded]

print("-_-_-_-_-_-")
print(input_data_encoded)

predicate_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicate_class)[0])

