import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def make_confusion_matrix(y_test, y_pred, classes, accuracy=None):
    print("Saving Confusion Matrix ... \n")
    cm = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')

    # Set plot labels
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    if accuracy:
        plt.title(f'Confusion Matrix for Classifier\nAccuracy: {accuracy:.3f}')
    else:
        plt.title('Confusion Matrix Heatmap')
    # Save the plot as a PNG file
    plt.savefig('./Decision Tree/Figures/confusion_matrix.png')
