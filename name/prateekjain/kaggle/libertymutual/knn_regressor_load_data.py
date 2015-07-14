from __future__ import division
import logging
import ast
from pprint import pprint
import itertools
import math
import numpy

from pandas import read_csv, pandas, DataFrame
from sklearn import cross_validation, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

logging.basicConfig(filename='knn_liberty_mutual_regression.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
google_file_location = ''

csv.field_size_limit(sys.maxsize)


output_file = open('knn_liberty_mutual.csv','w')
output_file_ml = open('liberty_mutual_ml.csv','w')

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


def one_hot_column(df, cols, vocabs):
    mats = [];
    df2 = df.drop(cols, axis=1)
    #print sps.lil_matrix(np.array(df2))
    #mats.append(sps.lil_matrix(np.array(df2)))
    for i, col in enumerate(cols):
        mat = sps.lil_matrix((len(df), len(vocabs[i])))
        for j, val in enumerate(np.array(df[col])):
            mat[j, vocabs[i][val]] = 1.
        mats.append(mat)

    res = sps.hstack(mats)
    return res



def f(x):
   math.log(x,2)



'''
Picking the verticals and content column from the data frame. If additional features have to be added, add a column name here
'''
df = panda_data_frame[['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17','T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']]
values = panda_data_frame[['Hazard']]
df_id = panda_data_frame[['Id']]

test_df = testing_data_frame[['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17','T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']]
test_df_id = testing_data_frame['Id']


log_message('Size of data set : ' + str(len(panda_data_frame)))
log_message('Size of values set : ' + str(len(values)))


t1_v4_mapping = {
       'N': 3,
       'H': 2,
       'E': 1,
       'W':4,
       'B':5,
       'C':6,
       'S':7,
       'G':8}


t1_v5_mapping = {
       'K': 3,
       'H': 2,
       'E': 1,
       'B':4,
       'C':5,
       'I':6,
        'A':7,
        'D':8,
        'J':9,
        'L':10}


t1_v6_mapping = {
        'N':0,
        'Y':1}


t1_v7_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4}


t1_v8_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4}



t1_v9_mapping = {
        'B':1,
        'C':2,
        'D':3,
        'E':4,
        'F':5,
        'G':6}



t1_v11_mapping = {
       'K': 9,
       'H': 6,
       'E': 4,
        'F':5,
       'B':2,
       'I':7,
        'A':1,
        'D':3,
        'J':8,
        'L':10,
        'M':11,
        'N':12}


t1_v12_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4}



t1_v15_mapping = {
        'A':1,
        'C':2,
        'D':3,
        'F':4,
        'H':5,
        'N':6,
        'S':7,
        'W':8}



t1_v16_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4,
        'E':5,
        'F':6,
        'G':7,
        'H':8,
        'I':9,
        'J':10,
        'K':11,
        'L':12,
        'M':13,
        'N':14,
        'O':15,
        'P':16,
        'Q':17,
        'R':18}



t1_v17_mapping = {
        'N':0,
        'Y':1}



t2_v3_mapping = {
        'N':0,
        'Y':1}


t2_v5_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4,
        'E':5,
        'F':6}


t2_v11_mapping = {
        'N':0,
        'Y':1}


t2_v12_mapping = {
        'N':0,
        'Y':1}


t2_v13_mapping = {
        'A':1,
        'B':2,
        'C':3,
        'D':4,
        'E':5}



df['T1_V4'] = df['T1_V4'].map(t1_v4_mapping)
test_df['T1_V4'] = test_df['T1_V4'].map(t1_v4_mapping)

df['T1_V5'] = df['T1_V5'].map(t1_v5_mapping)
test_df['T1_V5'] = test_df['T1_V5'].map(t1_v5_mapping)

df['T1_V6'] = df['T1_V6'].map(t1_v6_mapping)
test_df['T1_V6'] = test_df['T1_V6'].map(t1_v6_mapping)

df['T1_V7'] = df['T1_V7'].map(t1_v7_mapping)
test_df['T1_V7'] = test_df['T1_V7'].map(t1_v7_mapping)


feats =df.transpose().to_dict().values()
test_feats = test_df.transpose().to_dict().values()

#

from sklearn.feature_extraction import DictVectorizer
Dvec = DictVectorizer()

df = Dvec.fit_transform(feats).toarray()
test_df = Dvec.fit_transform(test_feats).toarray()
# vocabs = []
# vals = ['N', 'H', 'E', 'W', 'B', 'C', 'S', 'G']
# vocabs.append(dict(itertools.izip(vals, range(len(vals)))))
# vals = ['B', 'K', 'H', 'C', 'I', 'A', 'D', 'J', 'E', 'L']
# vocabs.append(dict(itertools.izip(vals, range(len(vals)))))
#res = one_hot_column(panda_data_frame, ['T1_V6', 'T1_V8'],vocabs).todense(),  # T1_V6,T1_V7,T1_V8,T1_V9,T1_V11,T1_V12,T1_V15,T1_V16,T1_V17,T2_V3,T2_V5,T2_V11,T2_V12,T2_V13'], vocabs).todense()




#res = DataFrame(res.toarray(), columns=features, index=observations)

#print len(res)
#print len(values)

clf = KNeighborsRegressor(n_neighbors=5,weights='distance')
clf.fit(df,values)


Y_train = clf.predict(df)

Y = clf.predict(test_df)
print clf.score(df,values)
#output_file.write('Id,Hazard\n')

output_file.write('Actual,Predicted\n')

diff_count = 0


for id,predicted,actual in zip(list(df_id.values.flatten()),Y_train,list(values.values.flatten())):
    output_file.write(str(id)+','+str(predicted[0])+','+str(actual)+','+str(predicted[0]-actual)+','+str(int(predicted[0])-actual)+'\n')



#for id,predicted in zip(list(test_df_id.values.flatten()),Y_train):
#    output_file.write(str(id)+','+str(predicted)+'\n')



values = values.values.flatten()
values = numpy.delete(values,0)
print 'Gini value',str(normalized_gini(values,Y_train))

output_file.flush()
output_file.close()


