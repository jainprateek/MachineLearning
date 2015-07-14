
__author__ = 'prateek.jain'

import codecs
import itertools
import time
import csv
import sys
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

__author__ = 'prateek.jain'

#logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
logging.basicConfig(filename='create_data_dump.log', filemode='w', level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


csv.field_size_limit(sys.maxsize)

sep = b","
quote_char = b'"'

stop = stopwords.words('english')
porter = PorterStemmer()

os.chdir(".")
for file_name in glob.glob("*.txt"):
    if os.path.exists(file_name):
        os.remove(file_name)


text_rows = []

text_labels = []

positive_labels_sample = {}

data_dump_location = '/Users/prateek.jain/work/datasets/dmoz'
all_files_data_dump_location = '/Users/prateek.jain/work/datasets/dmoz/all'

training_file_object = codecs.open('/Users/prateek.jain/work/python-workspace/webcrawler/file_training_combined.csv',
                                   'r', 'utf-8')
wr1 = csv.reader(training_file_object, dialect='excel', quotechar=quote_char, quoting=csv.QUOTE_ALL, delimiter=sep)

output_file = 'output.csv'
output_file_object = open(output_file, 'w')

dir_locations = []

if os.path.exists(all_files_data_dump_location):
    logging.info('Removing %s',all_files_data_dump_location)
    shutil.rmtree(all_files_data_dump_location)


logging.info('Creating %s',all_files_data_dump_location)
os.makedirs(all_files_data_dump_location)
logging.info('Finished creating %s',all_files_data_dump_location)




def get_path(label, sample_type):
    return data_dump_location + os.sep + label + os.sep + sample_type


def create_dir_put_files(label, file_path):
    pos_path = get_path(label,'pos')
    neg_path = get_path(label,'neg')
    if not pos_path in dir_locations:
        if os.path.exists(pos_path):
            logging.info('Removing %s',pos_path)
            shutil.rmtree(pos_path)
            logging.info('Finished Removing %s',pos_path)


        if os.path.exists(neg_path):
            logging.info('Removing %s',neg_path)
            shutil.rmtree(neg_path)
            logging.info('Finished Removing %s',neg_path)

        logging.info('Creating %s',pos_path)
        os.makedirs(pos_path)
        logging.info('Finished creating %s',pos_path)
        logging.info('Creating %s',neg_path)
        os.makedirs(neg_path)
        logging.info('Finished creating %s',neg_path)
        dir_locations.append(pos_path)
    put_file_in_dir(file_path, pos_path)
    #put_file_in_allfiles_dir(file_path, all_files_data_dump_location)
    return pos_path


def create_neg_dir_put_files(label, file_path, sample_type):
    path = data_dump_location + os.sep + label + os.sep + sample_type
    os.makedirs(path)
    for file_name in labels:
        put_file_in_dir(file_path, path)
        put_file_in_allfiles_dir(file_path, all_files_data_dump_location)


def put_file_in_dir(src, dir_location):
    copy(src, dir_location)


def put_file_in_allfiles_dir(label, all_files_data_dump_location):
    copy(label, all_files_data_dump_location)


counter = 0

for row in wr1:
    text_rows.append(row[6])
    labels = row[7].strip().split('|')
    empty_list = []

    file_name = str(counter) + '.txt'
    counter += 1
    temp_file = open(all_files_data_dump_location+os.sep+file_name, 'w')
    temp_file.write(row[6])
    temp_file.flush()
    temp_file.close()

    for label in labels:

        label = '_'.join(label.lower().split('/'))
        label = label.replace('_','', 1)
        if not ('http:' in label.lower() or 'www:' in label.lower()):
            if not label.lower() in text_labels:

                #label = label[1:]
                print 'Creating dir:'+label
                create_dir_put_files(label, temp_file.name)
                pos_sample_list = []
                pos_sample_list.append(counter)
                positive_labels_sample[label] = pos_sample_list
                text_labels.append(label.lower())
            else:
                pos_sample_list = positive_labels_sample[label]
                pos_sample_list.append(counter)
                positive_labels_sample[label] = pos_sample_list
                print 'Putting file ',temp_file,' inside ',get_path(label,'pos')
                put_file_in_dir(temp_file.name,get_path(label,'pos'))
            empty_list.append(label)
    text_labels.append(empty_list)


logging.info('Number of distinct files generated %d',counter)

file_name_set = set()

for value in range(counter):
    file_name_set.add(value)

logging.info('File name set generated %d',counter)

for label in positive_labels_sample.keys():
    pos_file_names = positive_labels_sample[label]
    pos_file_name_set = set(pos_file_names)
    logging.info('Generating files for label %s',label)
    logging.info('Positive file name set generated %s',str(pos_file_name_set))
    neg_set_diff = file_name_set.difference(pos_file_name_set)
    logging.info('Negative file name set generated %s',str(neg_set_diff))
    neg_dir_path = get_path(label, 'neg')
    for elements in neg_set_diff:
        file_location = all_files_data_dump_location + os.sep + str(elements)+'.txt'
        copy(file_location, neg_dir_path)
        logging.info('Copied negative file %s',file_location)

logging.info('Finished copying the negative files')
shutil.rmtree(all_files_data_dump_location)
logging.info('Deleting the files in the temporary directory location')

#os.chdir(".")
#for file_name in glob.glob("*.txt"):
#    os.remove(file_name)

#files = glob.glob('.')

#for f in files:
#    os.remove(f)
