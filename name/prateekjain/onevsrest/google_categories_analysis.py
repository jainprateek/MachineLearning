from __future__ import division
import logging
import ast

from pandas import read_csv
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

__author__ = 'prateek.jain'

import csv
import sys

from sklearn.preprocessing import MultiLabelBinarizer

__author__ = 'prateek.jain'

logging.basicConfig(filename='google_categories_analysis.log', filemode='w', level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
google_file_location = ''

csv.field_size_limit(sys.maxsize)


'''
Creating a panda data frame by reading the CSV file
'''
panda_data_frame = read_csv('/Users/prateek.jain/work/datasets/google_sites_100.csv',quotechar='"')



'''
Picking the verticals and content column from the data frame. If additional features have to be added, add a column name here
'''
df = panda_data_frame[['verticals','content']]


'''
Split training and testing 80% of the data goes for training, 20% will go
'''

train, test = train_test_split(df, test_size = 0.2)

numbers = []


def get_label_list(vertical_list):
    labels = []
    for vertical in vertical_list:
        #logging.info('Found vertical : %d',int(vertical[0]))
        labels.append(int(vertical[0]))
    return labels





def log_message(message):
    logging.info(message)




for idx, row in train.iterrows():
    if isinstance(row['verticals'], basestring):
        if row['verticals'].startswith('['):
            vertical_list = ast.literal_eval(row['verticals'])
            label_list = get_label_list(vertical_list)
            if len(label_list)==0:
                row['verticals']='del'
            else:
                row['verticals']=label_list
        else:
            row['verticals']='del'
    else:
        row['verticals']='del'
                #Remove this row




for idx, row in test.iterrows():
    if isinstance(row['verticals'], basestring):
        if row['verticals'].startswith('['):
            vertical_list = ast.literal_eval(row['verticals'])
            label_list = get_label_list(vertical_list)
            if len(label_list)==0:
                row['verticals']='del'
            else:
                row['verticals']=label_list
        else:
            row['verticals']='del'
    else:
        row['verticals']='del'





log_message('Before removing size of training set : '+str(len(train)))
train = train[train.verticals != 'del']
train = train.dropna()
#train = train[pandas.notnull(test['content'])]
log_message('After removing size of training set : '+str(len(train)))

log_message('Before removing size of test set : '+str(len(test)))
test = test[test.verticals != 'del']
test = test.dropna()
#test = test[pandas.notnull(test['content'])]
log_message('After removing size of test set : '+str(len(test)))

mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform(train.verticals)


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

training_data = train['content']
testing_labels = test['verticals']

print training_data.shape
print y_train_mlb.shape

classifier.fit(training_data,y_train_mlb)
#classifier.fit(train['content'],train['verticals'])
predicted = classifier.predict(test['content'])
all_labels = mlb.inverse_transform(predicted)
scores = cross_validation.cross_val_score(classifier, training_data, train.verticals, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

for item, labels in zip(test['content'], all_labels):
    #list_labels = ', '.join(labels)
    str_list_labels = str(labels)
    log_message('%s => %s' % (item,str_list_labels))