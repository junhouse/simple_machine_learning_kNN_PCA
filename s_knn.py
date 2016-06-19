#simple machine learning projects with Korean and Japanese money

%matplotlib inline
import numpy as np
import scipy as sp
import os, sys
from scipy import ndimage
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from PIL import Image


#width and height we are going to work with for the .png pictures
width = 322
height = 137
num_of_pics = 9
#paths for folders
#korean currency
path_k_1000 = 'Korea/1000/'
path_k_5000 = 'Korea/5000/'
path_k_10000 = 'Korea/10000/'
path_k_50000 = 'Korea/50000/'
#japanese currency
path_j_1000 = 'Japan/1000_yen/'
path_j_5000 = 'Japan/5000_yen/'
path_j_10000 = 'Japan/10000_yen/'
#list
paths = [path_k_1000, path_k_5000, path_k_10000, path_k_50000, path_j_1000, path_j_5000, path_j_10000]
file_name = ['1000_', '5000_', '10000_', '50000_', 'j_1000_', 'j_5000_', 'j_10000_']


#TESTING 
img = Image.open('Korea/1000/1000_9.png') # image extension *.png,*.jpg
new_width  = 322
new_height = 187
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save('test.png') # format may what u want ,*.png,*jpg,*.gif
string = str(10)
type(string)
file_name


#function that takes image and converts it into one dimensional array and append
def img_to_array(array, img_path):
    #open image and convert
    png = Image.open(img_path)
    png.load()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    background
    test = np.array(background)
    test = test.flatten()
    #use numpy stack 
    array = np.vstack((array, test))
    return array



#now we want to normalizae all the .png data into same size. So we will have same number of features
#there are total of 9 * 7 = 63 images, and each folder has 9 images
index = 0
data = np.zeros(132342) #give initial first row, and get rid of this row later
for path in paths:
    #if path is Korean currency
    if path[0] == 'K':
        num_pics = 9
        start = 0
        while start < num_pics:
            #get the path for the image
            pic_num = start + 1
            path_image = path + file_name[index] + str(pic_num) + '.png'
            new_path_image = path + file_name[index] + str(pic_num)+'_N' + '.png'
            img = Image.open(path_image)
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(new_path_image)
            #insert this information into the numpy array
            data = img_to_array(data,new_path_image)
            #update index
            start = start + 1
    #if path is Japanese currency
    elif path[0] == 'J':
        num_pics = 9
        start = 0
        while start < num_pics:
            #get the path for the image
            pic_num = start + 1
            path_image = path + file_name[index] + str(pic_num) + '.png'
            new_path_image = path + file_name[index] + str(pic_num)+'_N' + '.png'
            img = Image.open(path_image)
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(new_path_image)
            #insert this information into the numpy array
            data = img_to_array(data,new_path_image)
            #update index
            start = start + 1
    index = index + 1
data = np.delete(data, 0, 0)


#Displaying image from array -> picture
STANDARD_SIZE = (322, 137)#standardized pixels in image.
def get_image(mat):
    size = STANDARD_SIZE[0]*STANDARD_SIZE[1]*3
    r,g,b = mat[0:size:3], mat[1:size:3],mat[2:size:3]
    rgbArray = np.zeros((STANDARD_SIZE[1],STANDARD_SIZE[0], 3), 'uint8')#3 channels
    rgbArray[..., 0] = r.reshape((STANDARD_SIZE[1], STANDARD_SIZE[0]))
    rgbArray[..., 1] = b.reshape((STANDARD_SIZE[1], STANDARD_SIZE[0]))
    rgbArray[..., 2] = g.reshape((STANDARD_SIZE[1], STANDARD_SIZE[0]))
    return rgbArray

def display_image(mat):
    with sns.axes_style("white"):
        plt.imshow(get_image(mat))
        plt.xticks([])
        plt.yticks([])


display_image(data[0])
#example of displying an image

#Machine learning part (PCA and kNN)

from sklearn.decomposition import PCA
pca = PCA(n_components=60)
X = pca.fit_transform(data)
print pca.explained_variance_ratio_.sum()
pca

#Above code tells us that 99percent of variance is explained by 60 features. Let's create pandas dataframe with all the labels, and features
#Create DataFrame
DF = pd.DataFrame(columns={"Label"})
type_bills = np.empty([63,], dtype = '|S24')
type_KJ = ['K_1000', 'K_5000', 'K_10000', 'K_50000', 'J_1000', 'J_5000', 'J_10000']
start = 0
stop = 9
count = 0
while count < 63:
    type_bills[count] = type_KJ[count/9]
    count = count + 1

#save each pc value to coressponding variable
df = pd.DataFrame({"Label":type_bills})
for i in range(pca.explained_variance_ratio_.shape[0]):
    df["pc%i" % (i+1)] = X[:,i]
df.head()

#function that will help displaying component of PCA
def normit(a):
    a=(a - a.min())/(a.max() -a.min())
    a=a*256
    return np.round(a)
def getNC(pc, j):
    size=322*137*3
    r=pc.components_[j][0:size:3]
    g=pc.components_[j][1:size:3]
    b=pc.components_[j][2:size:3]
    r=normit(r)
    g=normit(g)
    b=normit(b)
    return r,g,b
def display_component(pc, j):
    r,g,b = getNC(pc,j)
    rgbArray = np.zeros((137,322,3), 'uint8')
    rgbArray[..., 0] = r.reshape(137,322)
    rgbArray[..., 1] = g.reshape(137,322)
    rgbArray[..., 2] = b.reshape(137,322)
    plt.imshow(rgbArray)
    plt.xticks([])
    plt.yticks([])


#Testing
display_component(pca,23)

#Graph kNN 

k_mark = 'o'
j_mark = 's'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for label, color in zip(df['Label'].unique(), colors):
    mask = df['Label']==label
    if(label[0] == 'K'):
        plt.scatter(df[mask]['pc1'], df[mask]['pc2'], c=color, label=label, marker=k_mark, s = 150)
    elif(label[0] == 'J'):
        plt.scatter(df[mask]['pc1'], df[mask]['pc2'], c=color, label=label, marker=j_mark, s = 150)
plt.title("kNN between Korean currency and Japanese currency")
plt.legend()















