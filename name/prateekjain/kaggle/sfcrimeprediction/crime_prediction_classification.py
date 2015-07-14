from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.util.testing import Series, DataFrame
from scipy.spatial.distance import correlation
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC

__author__ = 'prateek.jain'


#category_mapping={}
#district_mapping={}

category_id = 0
district_id = 0


list_district=[]
list_category=[]

def read_csv_file(location):
    return pd.read_csv(location)




def read_training_file(location):
    return read_csv_file(location)



def read_test_file(location):
    return read_csv_file(location)

# def get_category_mapping(category):
#     if category not in category_mapping:
#         global category_id
#         category_mapping[category]=category_id
#         category_id =+1
#     return category_mapping[category]


# def get_district_mapping(district):
#     if district not in district_mapping:
#         global district_id
#         district_mapping[district]=district_id
#         district_id =+1
#     return district_mapping[district]
#


#def generate_predictions(dis)


if __name__=='__main__':

    file_header = ['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']

    data_frame = read_training_file('/Users/prateek.jain/work/datasets/kaggle-competition/sf-crime/train.csv')
    data_frame_test = read_training_file('/Users/prateek.jain/work/datasets/kaggle-competition/sf-crime/test.csv')


    labels =  data_frame['Category']
    pd_frame = data_frame['PdDistrict']
    resolution = data_frame['Resolution']
    data_frame.drop(['Category'],inplace=True,axis=1)
    #training_data = pd.concat([pd_frame,resolution], axis=1)
    training_data = data_frame.as_matrix(['Dates','DayOfWeek','Address'])
    testing_data = data_frame_test.as_matrix(['Dates','DayOfWeek','Address'])


    gnb = MultinomialNB(alpha=0)
    #gnb = LinearSVC()

    print 'Made it till here-1'
    fh = FeatureHasher(input_type='string',non_negative=True)
    X=fh.fit_transform(training_data)
    X_test = fh.fit_transform(testing_data)


    print 'Made it till here-2'
    print training_data.shape

    #print X.toarray()
    print 'Made it till here-3'

    gnb_model = gnb.fit(X,labels)
    y_pred=gnb_model.predict(X_test)

    print len(y_pred)

    #for actual,predicted in zip(labels,y_pred):
    #    print actual,'=>',predicted


    crime_type_dict={}

    for type in file_header:
        value=[]
        crime_type_dict[type]=value


    for predicted in y_pred:
        for key in crime_type_dict.keys():
            if key!=predicted:
                zero_append = crime_type_dict[key]
                zero_append.append(0)
                crime_type_dict[key] = zero_append
            else:
                one_append = crime_type_dict[key]
                one_append.append(1)
                crime_type_dict[key] = one_append


    output = DataFrame(crime_type_dict)
    #output.index += 1
    output.to_csv('output_predict.csv',sep=',',index_label='Id')

    #print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(labels != y_pred).sum()))
    #s = Series(file_header)
    #correlation(training_data,labels)
    # for index, row in data_frame.iterrows():
    #     list_district.append(get_district_mapping(row['PdDistrict']))
    #     list_category.append(get_category_mapping(row['Category']))
    #
    # print 'Number of Districts',len(list_district)
    # print 'Number of Crimes',len(list_category)
    #
    # colors = cm.rainbow(np.linspace(0,1,len(list_district)))
    #
    # print 'Finished Processing Data.'
    # print 'Plotting Chart..Please Hold'
    #
    #
    # for x,y,c in zip(list_district,list_category,colors):
    #     plt.scatter(list_district,list_category)
    #
    #
    # plt.xlabel('Districts')
    # plt.ylabel('Crime Type')
    # plt.show()