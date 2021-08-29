#Fuzzy C means(FCM)
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import data, io, transform, color

plt.close('all')

def in_membfunction(clusters, data):
    #We define the initial membership function
    memberfunc = np.zeros([len(data), clusters])
    #Taking random values, where the sum of all clusters in a point equals to 1
    memberfunc = np.random.dirichlet(np.ones(clusters), len(data))
    return memberfunc

def calCentroids(memberfunc, ima1, m):
    #We calculate the the centroids
    centro = np.zeros([1,memberfunc.shape[1]])
    #Following the equation vik = sum(Uik**m * xki)/sum(Uik**m)    where i = cluster and k = the point xi
    for i in range(memberfunc.shape[1]):
        num = 0
        den = 0
        for k in range(memberfunc.shape[0]):
            aux = (memberfunc[k,i]**m)
            den = den + aux 
        for k in range(memberfunc.shape[0]):
            aux2 = (memberfunc[k,i]**m)*(ima1[k])
            num = num + aux2
        centro[0,i] = num/den     
    return centro


def newU(U, ima, centro,m):
    #The new membership function is calculate with the following equation
    #Unew = (sum(dik/djk)**(1/(m-1)))**-1
    distancias = np.zeros(U.shape)
    Unew = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for c in range(U.shape[1]):
            #We first calculate the eucladian distance of each pixel with each cluster centroid
            dis = np.sqrt((ima1[i] - centro[:,c])**2)
            #We store it in a matrix called 'distancias'
            distancias[i,c] = dis
            #In the following for loops is the tricky part
            #Each eucladian distance between a pixel and a cluster will be 
            #Divided with the sum of the euclidian distance between the same pixel and all cluster centroids
        for n in range(U.shape[1]):
            r = 0
            for j in range(U.shape[1]):
                el = (distancias[i,n] / distancias[i,j])**m
                r = r + el
            
            Unew [i,n] = (r)**-1
    return Unew

ima = data.camera()

clusters = 5 #Number of clusters in the image

m = 2 #weighting parameter

ima1 = ima.flatten() #In order to work with a column vector from the image

U = in_membfunction(clusters,ima1) #Initial membership function
for i in range(10): #Number of iterations 
    centro = calCentroids(U,ima1,m)
    Unew = newU(U,ima1,centro,m)
    U = Unew

#Finally we print the centroids gray level in the image pixels, in order to segment
cluster = np.zeros(ima1.shape)
for i in range(Unew.shape[0]):
    cluster[i] = centro[0,np.argmax(Unew[i])]

cluster = np.array(cluster).reshape(ima.shape)
plt.figure()
plt.imshow(cluster, cmap = 'gray')
plt.show()








           
    