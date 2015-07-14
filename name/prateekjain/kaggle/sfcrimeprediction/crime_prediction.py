from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import correlation
from sklearn import preprocessing, linear_model
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
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

    data_frame = read_training_file('/Users/prateek.jain/work/datasets/kaggle-competition/sf-crime/train.csv')

    labels =  data_frame['Category']
    pd_frame = data_frame['PdDistrict']
    resolution = data_frame['Resolution']
    data_frame.drop(['Category'],inplace=True,axis=1)
    training_data = pd.concat([pd_frame,resolution], axis=1)
    training_data = data_frame.as_matrix(['PdDistrict','Address'])
    regr = linear_model.LinearRegression()
    #gnb = LinearSVC()

    print 'Made it till here-1'
    fh = FeatureHasher(input_type='string',non_negative=True)
    X=fh.fit_transform(training_data)

    fhy = FeatureHasher(input_type='string',non_negative=True)
    Y = fhy.fit_transform(labels)


    knn_prediction = regr.fit(X,Y)
    print(regr.coef_)
    prediction = regr.predict(X)
    print regr.score(X, prediction)
    print 'Made it till here-2'
    print prediction

    #print X.toarray()
    #print 'Made it till here-3'

    #knn_prediction = knn.predict(X)
    print prediction
    #for actual,predicted in zip(labels,prediction):
    #    print actual,'=>',predicted

    print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(labels != knn_prediction).sum()))
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