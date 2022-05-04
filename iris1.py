import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import csv

C=3
training_data = []
testing_data = []


def get_data(split):
    training_data0 = []
    testing_data0 = []

    for i in range(C):
        filename = "class_"+str(i+1)+".txt"
        with open(filename,'r') as my_file:
            data = csv.reader(my_file, delimiter= ',')
            i = 0
            for line in data:

                if i < split:
                    training_data0.append(line)
                else:
                    testing_data0.append(line)
                i += 1
   
    for line in training_data0:
        
        line_new = []
        line_new.append(1)
        
        for value in line:
            line_new.append(float(value))

        training_data.append(line_new)

    for line in testing_data0:
        
        line_new = []
        line_new.append(1)
        
        for value in line:
            line_new.append(float(value))

        testing_data.append(line_new)
        
    return


def sigmoid(x, W):
    g = []
    z = np.dot(W.T,x)
    
    for i in z:
        g.append(1/(1+np.exp(-i)))
    return g

def iteration_W(W, grad_mse, alpha):
    
    W = W - alpha*grad_mse

    return W

def gradiant_mse(W, data, tn): #droppe types and features??
    
    grad_mse = np.zeros((5,3))
    
    for index, row in enumerate(data):

        x = row
        g = sigmoid(x, W)
        t = tn[index]

    
        mse_list = []
        
        for i in range(len(g)):
            term = np.multiply(np.multiply((g[i]-t[i]),g[i]), (1-g[i]))
            mse_list.append(term)

        mse_k = np.dot(np.array([x]).T,[mse_list]) 
        grad_mse = np.add(grad_mse, mse_k)

        return grad_mse
    
    
    
def get_mse(W, data, tn):
    
    mse_list = []

    for index, row in enumerate(data):
        
        x = row
        g = sigmoid(x, W)
        t = tn[index]
        
        term1 = np.dot(np.subtract(g, t).T, np.subtract(g, t))
        mse_list.append(term1)

    mse = 0
    
    for i in mse_list:
        mse += i

    mse = 0.5*mse

    return mse
   
    
    
def test(W, data, tn):
 
    # Counters for classified data
    correct = 0
    wrong = 0
    
    for index, row in enumerate(data):
        x = row
        g = sigmoid(x, W)
        t = tn[index]

        predicted_value = np.argmax(g)
        true_value  = np.argmax(t)
        
        if true_value == predicted_value:
            correct += 1
        else:
            wrong += 1
            

        error_rate = 100*wrong/(wrong + correct)  
    
    return error_rate


def confusion_matrix(W, data, tn):
    
    confusion_matrix = np.zeros((3, 3), dtype = int)
    
    for index, row in enumerate(data):
        x = row
        g = sigmoid(x, W)
        t = tn[index]

        predicted_value = np.argmax(g)
        true_value  = np.argmax(t)
        
        confusion_matrix[true_value][predicted_value] += 1
        
    return confusion_matrix


def run(split):
    
    get_data(split) #getting traning and testing data
    W = np.zeros((5,3))
    tn1 = [[1,0,0]]*30 + [[0,1,0]]*30 + [[0,0,1]]*30
    tn2 = [[1,0,0]]*20 + [[0,1,0]]*20 + [[0,0,1]]*20
    alpha = [0.01]
    iterations = 2
    

    confusion_train = []
    confusion_test = []

    for i in np.arange(0, 2000):
                
        grad_mse = gradiant_mse(W, training_data, tn1)
        W = iteration_W(W, grad_mse, alpha)
        error_train = test(W, training_data, tn1)

        mse = get_mse(W, training_data, tn1)

        error_test = test(W, testing_data, tn2)
       
    confusion_train.append(confusion_matrix(W, training_data, tn1))
    confusion_test.append(confusion_matrix(W, testing_data, tn2))
        
    return error_train, error_test, confusion_train, confusion_test

def plot_data(split):
    error_train, error_test, confusion_matrix_train, confusion_matrix_test = run(split)
    print(confusion_matrix_train)
    print('Error rate for training set:', error_train)
    print('Error rate for test set:', error_test)


    print('Confusion matrix for training set:')
    print(confusion_matrix_train)
    disp1 = ConfusionMatrixDisplay(confusion_matrix_train[0], display_labels= ["Setosa", "Versicolour", "Virginica"])
    disp1.plot()
    
    print('Confusion matrix for test set:')
    print(confusion_matrix_test)
    disp = ConfusionMatrixDisplay(confusion_matrix_test[0], display_labels= ["Setosa", "Versicolour", "Virginica"])
    disp.plot()
    plt.show()

plot_data(30)