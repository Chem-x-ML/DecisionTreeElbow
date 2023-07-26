import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataset import get_split_data
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from utils import make_confusion_matrix, make_lineplot
import sys

MAX_GROUP_TEST = 7
accuracies = []

f = open('./Decision Tree/log.txt', 'w+')
sys.stdout = f

for o in [True, False]:
    print(" ================================ \n")
    print(f'Running with overlap = {o} ... \n')
    OVERLAP = o
    for i in range(2, MAX_GROUP_TEST + 1):
        
        print(f'Testing with {i} functional groups ... \n')
            
        X_train, X_test, y_train, y_test, classes = get_split_data(overlap=OVERLAP, n_fgroups=i)

        clf = DecisionTreeClassifier()

        print("Training Model ...\n")
        clf.fit(X_train, y_train)

        print("Testing Model ... \n")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        print(f'Accuracy of the model is: {accuracy}\n')

        # convert one hot tensors to 1d labels
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        make_confusion_matrix(y_test, y_pred, classes, accuracy=accuracy, overlap=OVERLAP, n_fgroups=i)
     
    make_lineplot(accs=accuracies,overlap=OVERLAP)

print("Done!")
f.close()