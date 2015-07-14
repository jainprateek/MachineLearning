# -*- coding: utf-8 -*-
import codecs
import csv
import os
import sys
import itertools
import gc
import psutil as psutil
from sklearn.linear_model import SGDClassifier

csv.field_size_limit(sys.maxsize)

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

# from dense_transformer import DenseTransformer
import numpy as np


# data = df[['feature1', 'feature2']].values
sep = b","
quote_char = b'"'
# training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/dmoz_classification.csv'
training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv'
#training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/file_transformat'
testing_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/not_in_dmoz_website.csv'

training_file_object = codecs.open(training_file_path, 'r', 'utf-8')
testing_file_object = codecs.open(testing_file_path, 'r', 'utf-8')


# training_file_object = codecs.open(training_file_path,'r','ascii')
# testing_file_object = codecs.open(testing_file_path,'r','ascii')

output_file = 'output.csv'

wr1 = csv.reader(training_file_object, dialect='excel', quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)

wr2 = csv.reader(testing_file_object, dialect='excel', quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)

output_file_object = open(output_file, 'w')

text_rows = []

text_labels = []

test_rows = []

# lb = preprocessing.MultiLabelBinarizer()



def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


for row in wr1:
    text_rows.append(row[6])
    labels = row[7].strip().split('|')
    empty_list = []
    for label in labels:
        if not ('http:' in label.lower() or 'www:' in label.lower()):
            empty_list.append(label)
    text_labels.append(empty_list)

# For all the labels in the system
merged = list(itertools.chain(*text_labels))

my_set = set(merged)
class_label_list = list(my_set)
#class_label_list = [0,1]
all_class_labels = np.array(class_label_list)
# all_class_labels_transformed = lb.fit_transform(all_class_labels)

# For a label for each row
y_train_text = np.array(text_labels)

# labels_transformed = lb.fit_transform(y_train_text)

print("Did I make it till here?")

for row in wr2:
    test_rows.append(row[6])

X_train = np.array(text_rows)

# Y = lb.fit_transform(y_train_text)


X_test = np.array(test_rows)

y_train_text_chunks = chunks(y_train_text, 1000)

y_values = []

for chunk in y_train_text_chunks:
    y_values.append(chunk)

vectorizer = HashingVectorizer(decode_error='ignore',
                               non_negative=True)

clf = MultinomialNB(alpha=0.01)


clf = SGDClassifier(loss='log')

i = 0
for chunk in chunks(X_train,1000):
    Y = y_values[i]
    Y_array = np.array(Y)

    # Y_transformed = lb.fit_transform(Y_array)
    # Y = y_values[i]
    i = i + 1
    print('Training....')
    process = psutil.Process(os.getpid())
    print process.get_memory_info()[0] / float(2 ** 20)
    # vectorizer = CountVectorizer()
    x_train_transformed = vectorizer.transform(chunk)
    gc.collect()
    clf.partial_fit(x_train_transformed, Y_array, all_class_labels)
    gc.collect()


# gc.collect()


# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(lowercase=True,stop_words='english', )),
#     ('tfidf', TfidfTransformer()),
#     ('to_dense', DenseTransformer()),
#     ('clf', OneVsRestClassifier(tree.DecisionTreeClassifier()))])


# classifier = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('to_dense', DenseTransformer()),
#     ('clf', OneVsRestClassifier(tree.DecisionTreeClassifier()))])

transformed_test = vectorizer.transform(X_test)
predicted = clf.predict(transformed_test)
results = clf.predict_proba(transformed_test)

# all_labels = lb.inverse_transform(predicted)
# results = clf.predict_proba(X_test)[0]


# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(all_class_labels, results))

# gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
# results_ordered_by_probability = map(lambda x: x[0], sorted(zip(all_classes, results), key=lambda x: x[1], reverse=True))

# print results_ordered_by_probability

# for item, labels, probability in zip(X_test, all_labels,predicted_probability):
#     #print '%s => %s, %s' % (item, ', '.join(labels),str(probability))
#     output_file_object.write('%s => %s, %s' % (item, ', '.join(labels),str(probability))+'\n')


# , labels, probability in zip(X_test, all_labels,predicted_probability):
#     print '%s => %s, %f' % (item, ', '.join(labels))
#     output_file_object.write('%s => %s' % (item, ', '.join(labels))+'\n')



for item, labels in zip(x_train_transformed, all_class_labels):
    print '%s => %s' % (item, str(all_class_labels))
    output_file_object.write('%s => %s' % (item, str(all_class_labels)) + '\n')
