import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataset import get_split_data
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from utils import make_confusion_matrix


X_train, X_test, y_train, y_test, classes = get_split_data(overlap=False)

clf = DecisionTreeClassifier()

print("Training Model ...\n")
clf.fit(X_train, y_train)

print("Testing Model ... \n")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of the model is: {accuracy}\n')

# TODO confusion matrix, experiment with overlap vs no overlap, iterate with different number of functional groups

# convert one hot tensors to 1d labels
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

make_confusion_matrix(y_test, y_pred, classes, accuracy)


# fig = sns.heatmap(cm)
# fig.savefig('confusion_matrix.png')
# print(cm.shape)