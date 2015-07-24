from __future__ import division
import logging
import ast
from pprint import pprint
import itertools
import math
import numpy

from pandas import read_csv, pandas, DataFrame
from sklearn import cross_validation, linear_model, grid_search, decomposition
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import logistic
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import csv
import sys
import scipy.sparse as sps
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

__author__ = 'prateek.jain'

logging.basicConfig(filename='liberty_mutual_regression.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
google_file_location = ''

csv.field_size_limit(sys.maxsize)



output_file_ensemble = open('ensemble_liberty_mutual.csv','w')

'''
Creating a panda data frame by reading the CSV file
'''
panda_data_frame = read_csv('/Users/prateek.jain/work/datasets/kaggle-competition/liberty-mutual/train.csv')

testing_data_frame = read_csv('/Users/prateek.jain/work/datasets/kaggle-competition/liberty-mutual/test.csv')

print testing_data_frame.shape


def gini(solution, submission):
    df = sorted(zip(solution, submission), key=lambda x : (x[1], x[0]),  reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = np.sum([float(x[0]) for x in df])
    cumPosFound = np.cumsum([float(x[0]) for x in df])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [l - r for l, r in zip(Lorentz, random)]
    return np.sum(Gini)


def normalized_gini(solution, submission):
    solution = solution[1:]
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

def log_message(message):
    logging.info(message)


'''
Picking the verticals and content column from the data frame. If additional features have to be added, add a column name here
'''
df = panda_data_frame[['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17','T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']]
df_id = panda_data_frame[['Id']]
values = panda_data_frame[['Hazard']]


test_df = testing_data_frame[['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17','T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']]
test_df_id = testing_data_frame['Id']


log_message('Size of data set : ' + str(len(panda_data_frame)))
log_message('Size of values set : ' + str(len(values)))






def replace_and_concat(column_name,df):
    replacement = pandas.get_dummies(df[column_name],prefix=column_name)
    replacement = replacement.set_index(df.index)
    df.drop(column_name, axis=1, inplace=True)
    df = pandas.concat([df,replacement],axis=1)
    return df


list_to_replace = ['T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17','T2_V3','T2_V4','T2_V5','T2_V6','T2_V8','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14']
#list_to_replace = ['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12','T1_V15','T1_V16','T1_V17','T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']


for element in list_to_replace:
    df = replace_and_concat(element,df)
    test_df = replace_and_concat(element,test_df)

print df.shape
print test_df.shape




df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')



def predict_using_rfr_pca():
    rfr_output_file = open('rfr_pca_liberty_mutual.csv','w')
    rfr_output_file.write('Id,Hazard\n')
    parameters = {'oob_score':(True,False,)}
    rfr = RandomForestRegressor(n_estimators=200,verbose=1,warm_start=True)
    pca = decomposition.PCA(copy=True, whiten=False)
    pipe = Pipeline(steps=[('pca', pca), ('random_forest_regressor', rfr)])

    #Cs = np.logspace(-4, 4, 3)
    parameters_pca = {}

    estimator = grid_search.GridSearchCV(pipe,parameters_pca)
    #estimator.fit(df, values)
    estimator.fit(df,values)
    Y = estimator.predict(test_df)
    for id,predicted in zip(list(test_df_id.values.flatten()),Y):
        rfr_output_file.write(str(id)+','+str(predicted)+'\n')




def predict_using_rfr():
    rfr_output_file = open('rfr_liberty_mutual.csv','w')
    rfr_output_file.write('Id,Hazard\n')
    parameters = {'oob_score':(True,False)}
    rfr = RandomForestRegressor(n_estimators=200,verbose=1,warm_start=True)
    clf = grid_search.GridSearchCV(rfr, parameters)
    clf.fit(df,values)
    Y = clf.predict(test_df)
    for id,predicted in zip(list(test_df_id.values.flatten()),Y):
        rfr_output_file.write(str(id)+','+str(predicted)+'\n')



def predict_using_knn():
    knn_output_file = open('knn_liberty_mutual.csv','w')
    knn_output_file.write('Id,Hazard\n')
    parameters = {'n_neighbors':(10,),'weights':('uniform','distance'),'algorithm':('auto',)}
    clf_knn = KNeighborsRegressor()
    clf = grid_search.GridSearchCV(clf_knn, parameters)
    clf.fit(df,values)
    Y_knn_train = clf.predict(df)
    Y_knn = clf.predict(test_df)
    score = []
    for element in Y_knn:
        score.append(element[0])

    for id,predicted in zip(list(test_df_id.values.flatten()),score):
        knn_output_file.write(str(id)+','+str(predicted)+'\n')



def predict_using_gbr():
    gbr_output_file = open('gbr_liberty_mutual.csv','w')
    gbr_output_file.write('Id,Hazard\n')
    parameters = {'n_estimators':(1000,),'loss' : ('lad',),'alpha':(0.9,0.95),'warm_start':(True,False)}
    clf_gbr = GradientBoostingRegressor(learning_rate=1.0,max_depth=20, random_state=1,subsample=0.5)
    clf = grid_search.GridSearchCV(clf_gbr, parameters)
    clf.fit(df, values)
    print clf.score(df,values)
    Y_gbr = clf.predict(test_df)
    for id,predicted in zip(list(test_df_id.values.flatten()),Y_gbr):
        gbr_output_file.write(str(id)+','+str(predicted)+'\n')






predict_using_rfr_pca()
# predict_using_rfr()
# predict_using_knn()
# predict_using_gbr()






#predictions = [x/2 for x in zip(score,Y)]

#for id,predicted in zip(list(test_df_id.values.flatten()),predictions):
#    output_file_ensemble.write(str(id)+','+str(predicted)+'\n')

#
#
#
# Y_knn_train = [value for element in Y_knn_train for value in element]
#
# ensemble_train_df = DataFrame(Y_train,Y_knn_train,Y_train_gbr)
# ensemble_test_df = DataFrame(Y,Y_knn,Y_gbr)
#
# clf = RandomForestRegressor(n_estimators=120,verbose=1,warm_start=True,oob_score=True)
# clf.fit(ensemble_train_df,values)
# Y_ensemble = clf.predict(ensemble_test_df)
#
# for id,predicted in zip(list(test_df_id.values.flatten()),Y_ensemble):
#     output_file_ensemble.write(str(id)+','+str(predicted)+'\n')
#
#
#
# values = values.values.flatten()
# values = numpy.delete(values,0)
# print 'Gini value',str(normalized_gini(values,Y_train))
#
# output_file.flush()
# output_file.close()


