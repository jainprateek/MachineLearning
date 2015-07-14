from __future__ import division
import logging
import ast
import numpy

from pandas import read_csv
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import csv
import sys

from sklearn.preprocessing import MultiLabelBinarizer

__author__ = 'prateek.jain'

logging.basicConfig(filename='google_categories_analysis_cross_validation.log', filemode='w', level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
google_file_location = ''

csv.field_size_limit(sys.maxsize)


'''
Creating a panda data frame by reading the CSV file
'''
panda_data_frame = read_csv('/Users/prateek.jain/work/datasets/google_sites_100.csv',quotechar='"')



'''
Picking the verticals and content column from the data frame. If additional features have to be added, add a column name here
'''
df = panda_data_frame[['metatag.keywords','verticals','content']]



'''
Split training and testing 80% of the data goes for training, 20% will go
'''


numbers = []


def get_label_list(vertical_list):
    labels = []
    for vertical in vertical_list:
        labels.append(int(vertical[0]))
    return labels





def log_message(message):
    logging.info(message)




for idx, row in df.iterrows():
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



log_message('Before removing size of data set : '+str(len(df)))
df = df[df.verticals != 'del']
df = df.dropna()
log_message('After removing size of training set : '+str(len(df)))


mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform(df['verticals'])
#print y_train_mlb

features = ['content', 'metatag.keywords']


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

#print list(df.columns.values)
#data_content = df['content'].values
#data_metatag = df['metatag.keywords'].values
#data = numpy.concatenate(data_content,data_metatag)
data = df[features]
print data.shape

labels = df['verticals']
print labels.shape
#labels = labels.reshape(96,1)




#data_frame = np.reshape(df[features],df[features].shape)
classifier.fit(data,df.verticals)
#scores = cross_validation.cross_val_score(classifier,data,y_train_mlb, cv=5,scoring='f1_weighted')
#print scores
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))