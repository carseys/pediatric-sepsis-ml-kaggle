import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random


def format_for_modeling(df):
    random.seed(10)
    
    df.reset_index()
    person_id_index = np.argmax([column.startswith('new_person_id') for column in df.columns])
    column_list = []
    [column_list.append(i) for i in range(1,person_id_index)]
    [column_list.append(i) for i in range(person_id_index+1,len(df.columns.values))]
    X = df.iloc[:,column_list].values
    y = df.iloc[:,0].values
    person_id = df.iloc[:,person_id_index].values
    column_names = df.columns.values[column_list]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    for train_x_index, test_x_index in gss.split(X=X,y=y,groups=person_id):
        X_train = X[train_x_index]
        X_test = X[test_x_index]
        y_train = y[train_x_index]
        y_test = y[test_x_index]
        person_id_train = person_id[train_x_index]
        person_id_test = person_id[test_x_index]
        
    formatted_data = {}
    formatted_data['X_train'] = X_train
    formatted_data['X_test'] = X_test
    formatted_data['y_train'] = y_train
    formatted_data['y_test'] = y_test
    formatted_data['person_id_train'] = person_id_train
    formatted_data['person_id_test'] = person_id_test

    return formatted_data, column_names

def make_pred_file(df: pd.DataFrame, preds: np.array):
    predictions = pd.DataFrame()
    person_id = df['uid'].copy().apply(lambda x: x[19:])
    datetime_val = df['uid'].copy().apply(lambda x: x[:19])
    person_id_datetime = person_id + '_' + datetime_val
    predictions['person_id_datetime'] = person_id_datetime
    predictions['SepsisLabel'] = preds[:,1]
    predictions.to_csv('sepsis_predictions.csv',index=False)
    return predictions