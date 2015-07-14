import codecs
import itertools
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
import time
import csv
import sys
import pandas as pd
import numpy as np
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords


__author__ = 'prateek.jain'


csv.field_size_limit(sys.maxsize)



#df = pd.read_csv('/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv')
#df.tail()



#np.random.seed(0)
#df = df.reindex(np.random.permutation(df.index))
#df[['Text', 'Category']].to_csv('/Users/prateek.jain/Desktop/shuffled_file_training_combined.csv', index=False)


sep = b","
quote_char = b'"'

stop = stopwords.words('english')
porter = PorterStemmer()

text_rows = []


text_labels = []




training_file_object = codecs.open('/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv', 'r', 'utf-8')
wr1 = csv.reader(training_file_object,dialect='excel',quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)

output_file = 'output.csv'
output_file_object = open(output_file, 'w')


for row in wr1:
    text_rows.append(row[6])
    labels = row[7].strip().split('|')
    empty_list = []
    for label in labels:
        if not ('http:' in label.lower() or 'www:' in label.lower()):
            empty_list.append(label)
    text_labels.append(empty_list)


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

#dialect='excel'
def stream_docs(path):
    training_file_object = codecs.open(path, 'r', 'utf-8')
    wr1 = csv.reader(training_file_object,dialect='excel',quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)
    print(wr1.next())
    for row in wr1:
        text, label = row[6], row[7]
        labels = label.split('|')
        empty_list = []
        for label in labels:
            if not ('http:' in label.lower() or 'www:' in label.lower()):
                empty_list.append(label)
        yield text, empty_list


def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
#
# vect = HashingVectorizer(decode_error='ignore',
#                          n_features=2 ** 10,
#                          preprocessor=None,
#                          lowercase=True,
#                          tokenizer=tokenizer,
#                          non_negative=True,)
#
#
vect = HashingVectorizer(decode_error='ignore',
                          n_features=2 ** 10,
                          preprocessor=None,
                          lowercase=True,
                          tokenizer=tokenizer,
                          non_negative=True,)

# vect = TfidfVectorizer(decode_error='ignore',
#                          min_df=9,
#                          max_features=20,
#                          stop_words ='english',
#                          smooth_idf=True,
#                          norm = 'l2',
#                          sublinear_tf=False,
#                          use_idf=True)

from sklearn.linear_model import SGDClassifier

clf = MultinomialNB()
#clf = SGDClassifier(alpha=0.001)
#clf = OneVsRestClassifier(LinearSVC(random_state=0))
#
# clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
# doc_stream = stream_docs(path='/Users/prateek.jain/Desktop/shuffled_movie_data.csv')
doc_stream = stream_docs(path='/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv')

import pyprind

#pbar = pyprind.ProgBar(45)




merged = list(itertools.chain(*text_labels))
my_set = set(merged)

class_label_list = list(my_set)
#class_label_list = [0,1]
all_class_labels = np.array(class_label_list)
mlb = MultiLabelBinarizer(all_class_labels)

X_test_text, y_test = get_minibatch(doc_stream, 1000)

X_test = vect.transform(X_test_text)

classes = np.array([0, 1])
tick = time.time()
accuracy = 0
total_fit_time = 0
n_train_pos=0
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    X_train_matrix = vect.fit_transform(X_train)
    y_train = mlb.fit_transform(y_train)
    print X_train_matrix.shape,' ',y_train.shape
    clf.partial_fit(X_train_matrix.toarray(), y_train, classes=all_class_labels)
    total_fit_time += time.time() - tick
    n_train = X_train_matrix.shape[0]
    n_train_pos += sum(y_train)
    tick = time.time()


predicted = clf.predict(X_test)
all_labels = predicted

#results = clf.predict_proba(X_test)
#prob_per_class_dictionary = dict(zip(all_class_labels, results))

#print str(prob_per_class_dictionary)

#results_ordered_by_probability = map(lambda x: x[0], sorted(zip(all_labels, results), key=lambda x: x[1], reverse=True))

#for result in results_ordered_by_probability:
#    print result

for item,labels in zip(X_train, all_labels):
    print '%s => %s' % (item, labels)
    output_file_object.write('%s => %s' % (item, labels) + '\n')
    #prediction_time = time.time() - tick

#X_test, y_test = get_minibatch(doc_stream, size=5000)
#X_test = vect.fit_transform(X_test)
#print('Accuracy: %.3f' % clf.score(X_test.toarray(), y_test))

#clf = clf.partial_fit(X_test.toarray(), y_test)
