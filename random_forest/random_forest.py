import numpy as np
import pandas as pd
import requests
from decision_tree import decision_tree_classifier



class random_forest_classifier(object):
    def __init__(self,max_depth,n_estimators,min_gini,min_samples):
        pass




def accuracy(data):
    dt = decision_tree_classifier()
    tree = dt.fit(data)
    print('<========= decision tree ===========>')
    print(dt.traverse())
    predictions = dt.predict(data[0:-1][0:-1])
    true_values = [row[-1] for row in data]
    return '{:1f}'.format(sum([t==p for t,p in zip(true_values, predictions)])/len(true_values) *100 ) + '% accuracy' 
 
 
def flower_to_id(value):
    if value == 'Iris-virginica':
        return 2
    if value == 'Iris-versicolor':
        return 0
    if value == 'Iris-setosa':
        return 1


def get_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    values = requests.get(url).content.decode('utf-8').split('\n')
    data_set = [val.split(',') for val in values]
    data_df = pd.DataFrame(data=data_set,columns=['sepal_length','sepal_width','petal_length','petal_width','flower'])
    data_df = data_df.dropna()
    data_df[['sepal_length','sepal_width','petal_length','petal_width']] = data_df[['sepal_length','sepal_width','petal_length','petal_width']].astype('float')
    data_df['flower'] = data_df['flower'].map(flower_to_id)
    return data_df
    
data = get_data()
print(accuracy(data.to_numpy()))