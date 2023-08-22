import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image as im

height = 40
width = 40

def resize(image):
    resizedImg = image.resize((height, width))
    return resizedImg

def FacesMatrix(directory, matrix):
    faceList = os.listdir(directory)
    faceList.pop(0)
    #print(faceList)
    for face in faceList:
        img = im.open(directory+'/'+face)
        resizedImage = resize(img)
        imgArray = np.asarray(resizedImage, dtype=np.uint8)
        flatArray = imgArray.flatten()
        flatArray = np.column_stack(flatArray) #
        #print(flatArray)
        matrix = np.concatenate((matrix, flatArray), axis=0)
        #print(matrix)
        #print(len(matrix), len(matrix[0]))
    return matrix

def standardize(matrix):
   stand = matrix
   mean = matrix.mean(axis=(0), keepdims=False)
   sd = matrix.std(axis=(0), keepdims=False, ddof = 1)
   stand = (stand-mean) / sd
   return stand, matrix.mean(axis=(0), keepdims=False), matrix.std(axis=(0), keepdims=False, ddof=1)

def pca(matrix, dims):
    n, m = matrix.shape
    assert np.allclose(matrix.mean(axis=0), np.zeros(m))
    cov = np.cov(matrix, rowvar=0, ddof=1)
    eigenVals, eigenVects = np.linalg.eig(cov)

    idx = eigenVals.argsort()[-dims:][::-1]
    eigenVals = eigenVals[idx]
    eigenVects = eigenVects[:,idx]

    matrixpca = np.dot(-matrix, eigenVects)
    return matrixpca, eigenVals, eigenVects

def lossyCompression(stdmatrix, index, k, standardizedMean,standardizedSd):

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width, height),isColor=False)

    matrix, eigenVals, eigenVects = pca(stdmatrix,k)
   
    
    image = stdmatrix[[index], :]    

    for cols in range(k):
        features = eigenVects[:,:cols]
        
        proj = np.matmul(image,features)

        reconstruct = np.dot(proj, features.transpose())
       
        reconstruct *= standardizedSd
        reconstruct += standardizedMean

        reconstruct = reconstruct.reshape((40, 40))
        reconstruct[reconstruct>255] = 255
       
        out.write(np.uint8(reconstruct))

    out.release()
    cv2.destroyAllWindows()

def main():
    global height
    global width
    empty1DMatrix = np.empty([0, 1600], dtype=np.uint8)
    facesMatrix = FacesMatrix('yalefaces', empty1DMatrix)
    standardizedMatrix, standardizedMean, standardizedSd = standardize(facesMatrix)
    matrixpca, eigenVals, eigenVects= pca(standardizedMatrix, 2)
    plt.scatter(matrixpca[ : , 0],matrixpca[ : , 1]) 
    plt.savefig('plot.png')
    lossyCompression(standardizedMatrix, 0, 1600, standardizedMean, standardizedSd)

if __name__ == "__main__" :
    main()