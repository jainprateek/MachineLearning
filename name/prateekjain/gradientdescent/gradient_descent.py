__author__ = 'prateek.jain'

import numpy as np
from scipy import stats
import pylab


alpha = 0.01


def gradient_descent(x, y, numIterations):
    m = x.shape[0]
    theta = np.ones(2)
    print theta
    x_transpose = x.transpose()
    print m
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        #print iter,hypothesis
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        #print "iter %s | J: %.3f" % (iter, J)
        gradient = np.dot(x_transpose, loss) / m
        theta = theta - alpha * gradient  # update
    print theta
    return theta

'''
Function to generate random data for x,y
'''


def generate_data():
    """
     Returns a tuple for x,y co-ordinate points

     :rtype : tuple
     """
    #x = [1,2,3,4,5,6,7,8,9,10]
    #y = [1,2,3,4,5,6,7,8,9,10]

    x = np.random.random(1000)
    y = np.random.random(1000)
    return x, y


'''
Function to generate a linear regression
'''


def create_regression(x, y):
    """
    Returns a tuple for slope, intercept, r_value, p_value, std_err

    :rtype : tuple
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print "r-squared:", r_value**2
    return slope, intercept, r_value, p_value, std_err


if __name__ == '__main__':
    x, y = generate_data()
    linear_regression_tuple = create_regression(x, y)
    m = np.shape(x)
    x = np.c_[np.ones(m), x]  # insert column
    theta = gradient_descent(x, y, 10000)
    print theta

    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1]*x
        #print y_predict

    pylab.plot(x[:,1],y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    print "Done!"