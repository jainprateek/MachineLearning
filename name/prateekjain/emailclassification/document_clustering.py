import codecs
import csv
import re
from time import time
import nltk
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
import sys
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity

csv.field_size_limit(sys.maxsize)


__author__ = 'prateek.jain'

sep = b","
quote_char = b'"'
training_file_path = '/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv'
training_file_object = codecs.open(training_file_path, 'r', 'utf-8')
wr1 = csv.reader(training_file_object, dialect='excel', quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)

text_rows = []

text_labels = []


stemmer = SnowballStemmer("english")

for row in wr1:
    text_rows.append(row[6])
    labels = row[7].strip().split('|')
    empty_list = []
    for label in labels:
        if not ('http:' in label.lower() or 'www:' in label.lower()):
            empty_list.append(label)
    text_labels.append(empty_list)

true_k=2

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,init_size=1000,batch_size=1000, verbose=1)

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()

#vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
#                             stop_words='english')


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

X = vectorizer.fit_transform(text_rows)

#dist = 1 - cosine_similarity(X)

print(X.shape)
km.fit(X)

print "done in %0.3fs" % (time() - t0)
print

clusters = km.labels_.tolist()
print clusters

# print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
# print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
# print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
# print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_)
# print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, sample_size=1000)