'''
support vector machine classification algorithm.
they create a hyperplane which divides the data in a linear way. Eg plane, straight line
Find 2 closest points to the dividing line from either class and the distance between the 2 points and the line is the same
infinite hyperplanes can be generated.
we want the distances between the points and the line to be the maximum. This distance is called margin.
'''

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#print(x_train, y_train)
classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)
#clf=KNeighborsClassifier(n_neighbors=9)  COMPARING SVM AND KNEAREST NEIGHBORS
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
