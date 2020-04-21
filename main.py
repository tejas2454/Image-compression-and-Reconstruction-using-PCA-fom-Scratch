import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt

img1_org = img.imread('1.jpg') #reading the image
img2_org = img.imread('2.jpg') 
img3_org = img.imread('3.jpg') 
img1=img1_org[:,:,0] #picking 'B' matrix of RGB of the provided image
img2=img2_org[:,:,0]
img3=img3_org[:,:,0]

img1=(img1.reshape(1,img1.shape[0]*img1.shape[1])) #reshaping the matrix from PxP to 1xP^2
img2=(img2.reshape(1,img2.shape[0]*img2.shape[1]))
img3=(img3.reshape(1,img3.shape[0]*img3.shape[1]))

I=np.append(img1,img2,axis=0)
X=np.append(I,img3,axis=0) # appending the images 1xd to mxd matrix

centered_matrix = X - np.mean(X , axis = 0) #creating zero centered matrix i.e. the distribution of the image will be around zero-------------Point to remember
X1=np.cov(centered_matrix) #np.cov indicates the level to which 2 variables vary together ---- 1/m*summation(x-xi)---- creation of covariance matrix
eig_val, eig_vec = np.linalg.eigh(X1) #eigh for eigen values, eigh used to use properties of hermitian matrix

index_eigen_values_index = np.argsort(eig_val)#sorting in ascending order and returning the indexes 
index_eigen_sorted = index_eigen_values_index[::-1] # sorting in descending order

eig_vec = eig_vec[:,index_eigen_sorted] # rearranigng the eigen vectors in the same order as eigen values
eig_val = eig_val[index_eigen_sorted] # rearranging the eigen values matrix in the same order as eigen values arranged

k =1 # this can be changed by must be less than the p value or less than 
p = eig_vec.shape[1]
if k <p or k >0:
    eig_vec = eig_vec[:, range(k)] # taking eig_vec uptil the k value is defined ... this is the ''compressed version''
    
sigma = np.dot(eig_vec.T, centered_matrix) # centered matix =X matrix of PCA, W=eig_vec, U=eig_vec.T matrix of PCA. where as W and U are orthonormal to each other
recon = np.dot(eig_vec, sigma) + np.mean(X,axis=0).T #re-adding the mean done at line 21, to bring the matrixvalues to zero centered
recon_img_mat = np.uint8(np.absolute(recon)) # for avoid complex eigen values, thus obtaining the reconstructed eigen matrix

recon_img1=recon_img_mat[0].reshape((img1_org).shape[0],(img1_org).shape[1]) # reshaping the respective images to convert them into images
recon_img2=recon_img_mat[1].reshape((img2_org).shape[0],(img2_org).shape[1])
recon_img3=recon_img_mat[2].reshape((img3_org).shape[0],(img3_org).shape[1])

plt.imshow(recon_img1,cmap='gray') #showing the reconstructed arrays to images
print(recon_img1.shape)

#the sqared distance between the original and recovered vector
sumation=0
for i in range (centered_matrix.shape[0]):
#for j in range (centered_matrix.shape[1]):
    sumation=(X[i]-recon_img_mat[i])**2
np.argmin(sumation)

#recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMA