import os
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import glob
import cv2



#constants
random.seed(0)
e = 2**(-23)


#Data needs to be in the SAME DIRECTORY as python file
data = np.genfromtxt('diabetes.csv', delimiter=',')


#HW1 standardize
def standardize(data):
   stand = (data)
   test = data.mean(axis=(0), keepdims=False)
   test2 = data.std(axis=(0), keepdims=False)

   stand = (stand-test) / test2
   return stand

#Euclidean between a reference vector and an observation
def euclidean(centroid, observation):
    distance =math.sqrt(
        (np.power((centroid[0]-observation[0]),2))+ ((np.power((centroid[1]-observation[1]),2)))+ ((np.power((centroid[2]-observation[2]),2))))
    return distance


#Manhattan distance between old reference vector and new
def manhattan_distance(old, new):   
    temp_sum = 0
    for axis in range(len(old)):
        temp_sum  += abs((old[axis]) - (new[axis]))
    return temp_sum
   

##Function for updating the reference vectors. Takes the cluster and current ref vectors. Creates a new dictionary based on the Key and K
def update(cluster, centroids):
    updated_centroids = dict()
    
    for key in centroids.keys():
        obs = []
        ref = []
        cluster_data = [x[1] for x in cluster[key]]
        
        for axis in range(len(centroids[1])):
            for val in range(len(cluster_data)):
                obs.append(cluster_data[val][axis])
            mean =  sum(obs)/len(obs)
            ref.append(mean)
        updated_centroids.update({key: ref})

    return updated_centroids


# Calculates the clustering for each observation by calculating the Euclid. dist. and then assigning values to the min dist at a specific index.
def clustering(data, centroids):
    clusters = {
            i: []
            for i in centroids.keys()
        }
    for obs in range(np.size(data,axis=0)): 
        min_dist = 10000
        min_index = 0
        for i in centroids.keys():
            
            dist = euclidean(centroids[i], data[obs])
            if dist < min_dist: 
               
                min_dist = dist
                min_index = i
        clusters[min_index].append((obs, data[obs]))
    
    return clusters

#Plotting using matplot lib. Cycling through colors. Making video
def plot(clusters,name,purity):
   
    if os.path.isdir('./tempImages') == False:
        dir = os.mkdir('./tempImages')
    else:
        dir = './tempImages/'
    fig = plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    ax = fig.add_subplot(111, projection='3d')
    ass = str(purity)
    ax.set_title(" Purity = " + ass)

    for i in clusters.keys():
        for vals in range(len(clusters[i])):
            ax.scatter(clusters[i][vals][1][0], clusters[i][vals][1][1], clusters[i][vals][1][2], c = colors[i], marker='o')
        
       
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(str(dir) + str(name) + '.jpg')
    plt.close()

def video():
    img_array = []
    os.chdir('./tempImages')
    
    for filename in glob.glob("*.jpg"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width, height),isColor=True)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()

# calculating purity by checking the Y labels
def purity(clusters, labels):
    calc = 0
    for keys in clusters.keys():
        one = 0
        negative_one = 0
        for obs in range(len(clusters[keys])):
            index = clusters[keys][obs][0]
            if labels[index] == -1:
                negative_one +=1
            else:
                one +=1
        maxi = (len(clusters[keys])*(max(one,negative_one))) / len(clusters[keys])
        calc += maxi
    calc = calc/(len(labels))


    return calc
    
def myKMeans(X, target_clusterY, k):
    
    if np.size(X,1) > 3:
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)

    centroids = {
        i+1: X[random.randrange(0,np.size(X,axis=0))]
        for i in range(k)
    }

    sums = 10
    iters = 0
    frames = 20

    while sums > e:
        if iters < frames:
            clusters_data = clustering(X, centroids )
            pure = purity(clusters_data, class_label_Y)
            plot(clusters_data, sums, pure)
            
            centroids_updated = update(clusters_data, centroids)
            sums = 0
            for key in centroids.keys():
                sums +=manhattan_distance(centroids[key], centroids_updated[key])

            centroids = centroids_updated
            iters += 1
            print(str.format("frames left: {}", (40-iters)))
        else:
            video()
           
            break

            
###Fitting the data
class_label_Y = data[:,0]
obs_data_X = standardize(data[:,1:])

