import numpy as np
import matplotlib.pyplot as plt
import functions as fnc
import datetime as dt
from sklearn.cluster import KMeans
from tensorflow import keras
from keras.datasets import mnist

#--------Loading datasets-----------------------------------------------
def load(chunk_num, train_chunk, num_img):
       
    (train_data, train_target), (test_data, test_targets) = mnist.load_data()

    train_data = np.asarray(train_data)
    train_target = np.asarray(train_target)
    
    test_data = np.asarray(test_data)[0:min(num_img, len(test_data, chunk_num))]
    test_targets = np.asarray(test_targets)[0:min(num_img, len(test_targets, chunk_num))]
#training
    num_train, dim_x, dim_y = train_data.shape
    train_setx = train_data.reshape(num_train,dim_x*dim_y)
#testing
    num_test, dim_testx, dim_testy = test_data.shape
    test_data_reshaped = test_data.reshape(num_test,dim_testx*dim_testy)
#split
    train_data_split = np.asarray(np.split(train_setx))
    train_target_split = np.asarray(np.split(train_target))

    return train_data_split[train_chunk], train_target_split[train_chunk], test_data_reshaped[0:100], test_target[0:100]

#--------Calculating the Euclidean Distance between two vectors---------

def euclidean_dist(img1,img2):
    dist = (sum(img1 - img2)**2)
    return dist

def KNNClustering(clusters, test_pic, M):
    flattened_pic = test_pic.flatten().reshape(1, 28*28)
    distances = np.zeros(len(clusters))

    for i in range(len(clusters)):
        distances[i] = np.linalg.norm(flattened_pic - clusters[i])
    NN_index = np.argmin(distances)

    return NN_index // 64

#-------Locating the nearest neighbors/most similar---------------------

def nearest_neighbors(train, test_row, neighbors):
    dist = list()
    for i in train:
        distance = euclidean_dist(test_row,i)
        dist.append(i, distance)
    dist.sort(key=lambda tup: tup[0])
        
    nearest_neighbor = [dist[i][0] for i in range(neighbors)]
    return nearest_neighbor

#-------Predicting a classification with the nearest neighbors-----------

def prediction(train, test_row, neighbors):
    neighbors = nearest_neighbors(train, test_row, neighbors)
    classi = []

    for i in neighbors:
        classi.append(i[-1])
        pred = max(classi, key = classi.count)
        return pred
    
#-------K-Nearest neighbour classifier-------------------------------------
def KNN(train_data, train_targets, test_data, k):
    num_predictions = len(test_data)
    predictions = [None]*num_predictions
    for i in range(num_predictions):
        pred = prediction(train_data, train_targets, test_data[i], k)
        predictions[i] = pred

    
#-------Confusion Matrix----------------------------------------------    
def confusion(pred, t):
    confusion = np.zeros((10,10))
    for i in range(len(pred)):
        target = int(t[i])
        prediction = int(pred[i])
        confusion[target][prediction] += 1
    return confusion   

#--------Error Rate-----------------------------------------------------
    
def errorRate(conf, count):
    err = 0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            if i != j:
                err += conf[i][j]
                rate = np.round((err/count)*100,2)
        return rate

#---------Sorting the classes-------------------------------------------
def sortClasses(memory_data, targets):
    numb_classes = np.zeros(10)
    sorted_memory_data = np.empty_like(memory_data)
    sorted_targets = np.argsort(targets)

    for i in range(len(targets)):
        numb_classes[targets[i]] += 1
        sorted_memory_data[i] = memory_data[sorted_targets[i]]

    return np.asarray(sorted_memory_data), numb_classes


#--------Defining cluster----------------------------------------------

def clustering(memory_data, targets, M):
    time_start = dt.datetime.now().replace(microsecond=0)
    
    sorted_memory_data, numb_classes = sortClasses(memory_data, targets)
    flattened_sorted_memory_data = sorted_memory_data.flatten().reshape(memory_data.shape[0], 28*28)
    clusters = np.empty((len(numb_classes), M, 28*28))
    start = 0
    end = 0

    for count, i in enumerate(numb_classes):
        end += i
        cluster = KMeans(n_clusters=M,random_state=0).fit(flattened_sorted_memory_data[int(start):int(end)]).cluster_centers_
        start = end
        clusters[count] = cluster

    time_end = dt.datetime.now().replace(microsecond=0)
    print("Clustering time: ", time_end-time_start)
    return clusters.flatten().reshape(len(numb_classes)*64, 28*28)

#---------Plot misclassified pixtures-----------------------------------
def plt_missclass(miss, imgSize=28):
    for i in range(miss.len):
        title = "Prediction: " + str(miss.pred[i]) + " But it was: " + str(miss.target[i])
        plt.title(title)
        plt.xlabel(str(miss.contenders[i]))
        plt.imshow(np.reshape(miss.image[i],(imgSize,imgSize)))
        plt.show()
        input("Press for next pic")

        
def main():
#set number for data chunk, train_chunk and num_img
    chunk_num = 0
    train_chunk = 0
    num_img = 0
    train_data, train_targets, test_data, test_targets = load(chunk_num, chunk_num, num_img)
    test_size = len(test_targets)
    predictions = np.zeros(test_size)

# Setting value for K
    k = 7

    startTime = dt.datetime.now().replace(microsecond=0)
    pred = fnc.KNN(train_data, train_targets, test_data, k)

#Printing confusion matrix
    confusion_matrix = fnc.confusion(pred, test_targets)
    print("Confusion matrix: \n", confusion_matrix)

#printing error rate
    errorRate = fnc.errorRate(confusion_matrix, test_size)
    print("Error rate: ", errorRate)

#giving the endtime
    endTime = dt.datetime.now().replace(microsecond=0)
    print("Classification time: ", endTime-startTime)






