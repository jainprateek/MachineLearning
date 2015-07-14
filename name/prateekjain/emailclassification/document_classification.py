import itertools
from sklearn.preprocessing import MultiLabelBinarizer

__author__ = 'prateek.jain'

# -*- coding: utf-8 -*-
import codecs
import csv
import sys

csv.field_size_limit(sys.maxsize)

import gc

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from dense_transformer import DenseTransformer

import numpy as np



#data = df[['feature1', 'feature2']].values
sep=b","
quote_char = b'"'
#training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/dmoz_classification.csv'
training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv'
testing_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/not_in_dmoz_website.csv'


training_file_object = codecs.open(training_file_path,'r','utf-8')
testing_file_object = codecs.open(testing_file_path,'r','utf-8')


#training_file_object = codecs.open(training_file_path,'r','ascii')
#testing_file_object = codecs.open(testing_file_path,'r','ascii')

output_file = 'output.csv'

wr1 = csv.reader(training_file_object, dialect='excel',quotechar=quote_char,quoting=csv.QUOTE_ALL,delimiter=sep)

wr2 = csv.reader(testing_file_object, dialect='excel',quotechar=quote_char,quoting=csv.QUOTE_ALL,delimiter=sep)

output_file_object = open(output_file,'w')

text_rows = []

text_labels = []

test_rows = []

for row in wr1:
    text_rows.append(row[6])
    text_labels.append(row[7].split('|'))

for row in wr2:
    #row_text_labels = []
    test_rows.append(row[6])

#X_train = text_labels.toarray()
print '1st print'
X_train = np.array(text_rows)
print '2nd print'
#X_train = X_train.toarray()
y_train_text = np.array(text_labels)

print '3rd print'

X_test = np.array(test_rows)
print '4th print'
    #temp_array = np.array([])
    #X_train = np.append(X_train,row[6])
    #label_list = row[7].split('|')
    #for label in label_list:
    #    temp_array = np.append(temp_array,label)
    #y_train_text = np.append(y_train_text,temp_array)

#print str(X_train)
#print str(y_train_text)

# X_train = np.array(["new york is a hell of a town",
#                     "new york was originally dutch",
#                     "the big apple is great",
#                     "new york is also called the big apple",
#                     "nyc is nice",
#                     "people abbreviate new york city as nyc",
#                     "the capital of great britain is london",
#                     "london is in the uk",
#                     "london is in england",
#                     "london is in great britain",
#                     "it rains a lot in london",
#                     "london hosts the british museum",
#                     "new york is great and so is london",
#                     "i like london better than new york"])
# y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
#                 ["new york"],["london"],["london"],["london"],["london"],
#                 ["london"],["london"],["new york","london"],["new york","london"]]
#
# X_test = np.array(['nice day in nyc',
#                    'welcome to london',
#                    'london is rainy',
#                    'it is raining in britian',
#                    'it is raining in britian and the big apple',
#                    'it is raining in britian and nyc',
#                    'hello welcome to new york. enjoy it here and london too'])
#target_names = ['New York', 'London']

merged = list(itertools.chain(*text_labels))
my_set = set(merged)

class_label_list = list(my_set)
all_class_labels = np.array(class_label_list)

mlb = MultiLabelBinarizer(all_class_labels)

print '5th print'
Y = mlb.fit_transform(y_train_text)
print '6th print'


classifier = Pipeline([
    ('vectorizer', CountVectorizer(lowercase=True,stop_words='english', )),
    ('tfidf', TfidfVectorizer(decode_error='ignore',
                         min_df=10,
                         max_features=2000,
                         stop_words ='english',
                         smooth_idf=True,
                         norm = 'l2',
                         sublinear_tf=False,
                         use_idf=True)),
    ('to_dense', DenseTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))])


# classifier = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('to_dense', DenseTransformer()),
#     ('clf', OneVsRestClassifier(tree.DecisionTreeClassifier()))])


print '7th print'

gc.collect()
classifier.fit(X_train, Y)

print '8th print'

predicted = classifier.predict(X_test)
predicted_probability = classifier.predict_proba(X_test)

all_labels = mlb.inverse_transform(predicted)
results = classifier.predict_proba(X_test)[0]


# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(all_labels, results))

# gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
results_ordered_by_probability = map(lambda x: x[0], sorted(zip(all_labels, results), key=lambda x: x[1], reverse=True))

print results_ordered_by_probability

# for item, labels, probability in zip(X_test, all_labels,predicted_probability):
#     #print '%s => %s, %s' % (item, ', '.join(labels),str(probability))
#     output_file_object.write('%s => %s, %s' % (item, ', '.join(labels),str(probability))+'\n')


# , labels, probability in zip(X_test, all_labels,predicted_probability):
#     print '%s => %s, %f' % (item, ', '.join(labels))
#     output_file_object.write('%s => %s' % (item, ', '.join(labels))+'\n')



for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))
    output_file_object.write('%s => %s' % (item, ', '.join(labels))+'\n')

