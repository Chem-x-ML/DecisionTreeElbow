import pandas as pd
from sklearn.model_selection import train_test_split
from fgroups import get_common_fgroups_df
import ast



def get_split_data(n_fgroups, overlap=True ):

    data_df = get_common_fgroups_df(n_fgroups)
    
    if (not overlap):
        # remove compounds with fg_label = 'Complex'
        og_len = len(data_df)
        data_df = data_df[data_df['fg_label'] != 'Complex']
        print(f'Removed {og_len - len(data_df)} complex compounds')

    logs = data_df['normalised final log']
    X = [ast.literal_eval(ls) for ls in logs]

    y = pd.get_dummies(data_df['fg_label']).values
    classes = pd.get_dummies(data_df['fg_label']).columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, classes