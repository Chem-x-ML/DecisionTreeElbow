import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_confusion_matrix(y_test, y_pred, classes, accuracy=None, overlap=False, n_fgroups=2):
    print("Saving Confusion Matrix ... \n")
    cm = confusion_matrix(y_test, y_pred, normalize='pred')
    # cm =  cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    conf_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_df, annot=True, fmt='.2f', cmap='Blues')

    # Set plot labels
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    if accuracy:
        plt.title(f'Confusion Matrix for Classifier for {n_fgroups} classes {"(with overlap)" if overlap else ""}\nAccuracy: {accuracy:.3f}')
    else:
        plt.title('Confusion Matrix Heatmap')
        
    # Save the plot as a PNG file
    plt.savefig(f'./Decision Tree/Figures/cm_{n_fgroups}classes_{"o" if overlap else ""}.png')


def make_lineplot(accs, overlap=False):
    print("Saving Line Plot ... \n")
    groups = range(2, len(accs) + 2)
    plt.figure(figsize=(10, 8))
    sns.lineplot(x=groups, y=accs)
    plt.xlabel('Number of Functional Groups')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Number of Functional Groups {"with overlap" if overlap else ""}')
    plt.xticks(groups)
    plt.savefig(f'./Decision Tree/Figures/acc_vs_nfgroups_{"o" if overlap else""}.png')
    