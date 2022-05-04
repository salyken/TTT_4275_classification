import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def hist():

    # Load data from each class
    data1 = np.loadtxt(open("class_1", "rb"), delimiter=",", skiprows=1)
    data2 = np.loadtxt(open("class_2", "rb"), delimiter=",", skiprows=1)
    data3 = np.loadtxt(open("class_3", "rb"), delimiter=",", skiprows=1)

    # Plot histogram
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 6))
 
    #colors = ['pink', 'cayan', 'grey']
    
    #sepal_width
    ax0.hist(data1[:, 0],bins = 20, alpha = 0.45)
    ax0.hist(data2[:, 0],bins = 20, alpha = 0.45)
    ax0.hist(data3[:, 0],bins = 20, alpha = 0.45)
    ax0.legend(['Iris-setosa', 'Iris-versicolour', 'Iris-virginica'],prop={'size': 10})
    ax0.set_title('Sepal width [cm]')
    ax0.set_ylabel('Count')
    
    #sepal_length
    ax1.hist(data1[:, 1], bins = 20, alpha = 0.45)
    ax1.hist(data2[:, 1], bins = 20, alpha = 0.45)
    ax1.hist(data3[:, 1],bins = 20, alpha = 0.45)
    ax1.set_title('Sepal lengt [cm]')
    ax1.set_ylabel('Count')
    
    #petal_width
    ax2.hist(data1[:, 2],bins = 20, alpha = 0.45)
    ax2.hist(data2[:, 2],bins = 20, alpha = 0.45)
    ax2.hist(data3[:, 2],bins = 20, alpha = 0.45)
    ax2.set_title('Petal width [cm]')
    ax2.set_ylabel('Count')
    
    #petal_length
    ax3.hist(data1[:, 3],bins = 20, alpha = 0.45)
    ax3.hist(data2[:, 3],bins = 20, alpha = 0.45)
    ax3.hist(data3[:, 3],bins = 20, alpha = 0.45)
    ax3.set_title('Petal length [cm]')
    ax3.set_ylabel('Count')
    
    plt.show()
    


    
#to plot hist
#    hist()