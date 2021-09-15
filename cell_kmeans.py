# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:22:33 2021

@author: NING MEI
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import metrics
from sklearn.cluster import KMeans


class ImageDataset(): #读取dataset，并且把dataset的每张图变成28*28的黑白图像，并且把图像归一化成0，1点；
    def __init__(self,path_feature,path_label,size_x,size_y,x_max):
        self.X = self.load_datafromexcel(path_feature).astype('float32')/x_max
        self.Y = self.load_datafromexcel(path_label).astype('float32')/x_max
        self.X_tensor = self.df_to_tensor(self.X,size_x,size_y)
        self.Y_tensor = self.df_to_tensor(self.Y,size_x,size_y)
               
    def load_datafromexcel(self,path):
        Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
        Sample = Sample_excel.values  #转成numpy数组
        return Sample

    def df_to_tensor(self,df,size_x,size_y):
        N = df.shape[0]
        data_tensor = torch.tensor(df)
        data_tensor=data_tensor.reshape(N,size_x,size_y)
        return data_tensor
    
    def sample(self,k):
        return self.X_tensor[k,:,:],self.Y_tensor[k,:,:]

class Imagecoordinate():
    def __init__(self,image_X,image_Y):
        self.X = self.image_coordinate(image_X)
        self.Y = self.image_coordinate(image_Y)
        
    def image_coordinate(self,image_X):
        coordinate = pd.DataFrame() # 两列分别代表有坐标的横纵坐标
        for i in range(image_X.shape[0]):
            for j in range(image_X.shape[1]):
                if image_X[i,j]>0:
                    temp_coordinate = np.array([i,j])
                    temp_coordinate = temp_coordinate.reshape(1,-1)
                    #print([i,j])
                    coordinate = coordinate.append(pd.DataFrame(temp_coordinate))
                    #print(coordinate)
        coordinate = coordinate.values  #转成numpy数组
        coordinate = coordinate[:,0:coordinate.shape[1]]
        return coordinate

class Generate_pic(): #生成一张图
    def __init__(self,x_max,x_th,alpha0,beta0,step_end,X):
        self.x_max = x_max
        self.x_th = x_th
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.step_end = step_end
        self.X0 = X
        self.X = X
        self.Xend = self.generate_end()
        self.Xendpic = self.generate_pic()
        
    def alpha_step(self,i,j): #alpha影响的自身的增长函数
        delta = 0 #与Xij无关每次都不增长只能扩散
        delta = self.alpha0*self.X[i,j]
        return delta

    def beta_step(self,i,j): #其他菌落对单元增长的影响
        #左边界
        k_up = max([i-1,0])
        k_down = min([i+2,self.X.shape[0]])
        l_left = max([j-1,0])
        l_right = min([j+2,self.X.shape[1]])
        delta = 0
        for k in range(k_up,k_down): #元胞自动机
            for l in range(l_left,l_right):
                if k!=i or l!=j:
                    delta = delta - self.beta0/8*self.X[k,l]*(self.X[k,l]+self.X[i,j]-2*self.x_max+self.alpha0/self.beta0)
        return delta

    def generate_xijnew(self,i,j):
        delta_xij = (1-self.X[i,j]/self.x_max)*(self.alpha_step(i,j)+self.beta_step(i,j))
        if delta_xij<0:
            delta_xij = 0
        x_ijnew = self.X[i,j]+delta_xij
        if x_ijnew>self.x_max:
            x_ijnew = self.x_max
        return x_ijnew
    
    def generate_Xnew(self):
        X_temp = np.zeros(self.X.shape)
        for i in range(0,self.X.shape[0]):
            for j in range(0,self.X.shape[1]):
                X_temp[i,j] = self.generate_xijnew(i,j)
        return X_temp
    
    def generate_pic(self):
        X_temp = np.zeros(self.Xend.shape)
        for i in range(0,self.Xend.shape[0]):
            for j in range(0,self.Xend.shape[1]):
                if self.Xend[i,j]>=self.x_th:
                    X_temp[i,j]=255*self.Xend[i,j]/self.x_max
        return X_temp
    
    def generate_end(self):
        for t in range(0,self.step_end):
            X = self.generate_Xnew()
            self.X = X
        return X


def generate_image(image_coordinate,size_x,size_y,x_max): #输入坐标矩阵，输出image   
    image = np.zeros((size_x,size_y))
    for k in range(image_coordinate.shape[0]):
        print(image_coordinate[k,0],image_coordinate[k,1])
        image[image_coordinate[k,0],image_coordinate[k,1]]=x_max
    return image
    
def load_datafromexcel(path):
        Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
        Sample = Sample_excel.values  #转成numpy数组
        return Sample

if __name__ == '__main__':
    size_x=64 #图片的宽
    size_y=64 #图片的长
    x_max = 255 #图片中最大值
    x_th = 179 #能看到图案时的threshold
    alpha0 = 0.05 #公式中的alpha，代表自身增长率
    beta0 = 0.001 #公式中的beta，代表周围对齐影响
    step_end = 17#停止步骤，第150步停止；
    

    #第一步读取数据集
    #data_train = ImageDataset("data_X.xlsx","data_Y.xlsx",size_x,size_y,x_max) #
    
    #第二步选择一张图片实验
    #sample_num = 0 # 选择第0张图片
    sample_X= load_datafromexcel("data_Z.xlsx")
    sample_Y= load_datafromexcel("data_C.xlsx")
    plt.figure(1) #原始图片生长后
    plt.imshow(sample_X)
    plt.figure(2) #原始图片生长前
    plt.imshow(sample_Y)
    oneimage_coordinate = Imagecoordinate(sample_X,sample_Y) #把图片坐标化
    print(oneimage_coordinate.X)
    
    #第三步，开始执行K-means，猜测原始图片生长后是由K个点长出来的；
    K = 24 #初始点有几个
    clf = KMeans(n_clusters = K).fit(oneimage_coordinate.X)
    cluster_centers = np.rint(clf.cluster_centers_)
    cluster_centers = cluster_centers.astype(int)
    print(cluster_centers)
    sample_predictImageY = generate_image(cluster_centers,size_x,size_y,x_max)
    plt.figure(3) #用kmeans预测的原始状态
    plt.imshow(sample_predictImageY)
    
    #第四步，由预测值生成的新图片；
    pic_predict = Generate_pic(x_max=x_max,x_th=x_th,alpha0=alpha0,beta0=beta0,step_end =step_end,X = sample_predictImageY).Xendpic
    plt.figure(4)#由预测值生成的新图片；
    plt.imshow(pic_predict)
    
    