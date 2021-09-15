# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:51:06 2021
 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def load_picfromexcel(path):  #读取excel
    #统一格式为：没有header行，没有header列
    Sample_excel = pd.read_excel(path, header = None, index_col = None)
    Sample = Sample_excel.values  #转成numpy数组
    return Sample

def writedata_toexcel(data_X,path_X,data_Y,path_Y):
    data_X.to_excel(path_X)
    data_Y.to_excel(path_Y)
    return

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

def generate_randomX(size_x,size_y,p_max,x_max): #p_max是该点被滴到的概率
    X_temp = np.zeros((size_x,size_y))
    for i in range(0,X_temp.shape[0]):
        for j in range(0,X_temp.shape[1]):
            x_rand = random.random()
            if x_rand<=p_max:
                X_temp[i,j]=x_max
    return X_temp
 


class Data_mem():
    def __init__(self,size_x,size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.size = size_x*size_y
        self.col = np.arange(0,self.size,1)
        self.X = pd.DataFrame(columns = self.col)
        self.Y = pd.DataFrame(columns = self.col)
        
    def flatten_sample(self,X0,Xendpic):
        X0_flatten = X0.reshape(1,-1)
        Xendpic_flatten = Xendpic.reshape(1,-1)
        return X0_flatten,Xendpic_flatten        
    
    def add_sample(self,X0,Xendpic):
        X0_flatten,Xendpic_flatten = self.flatten_sample(X0,Xendpic)
        self.Y = self.Y.append(pd.DataFrame(X0_flatten))  #这里需要注意因为神经网络是逆向过程，所以可以倒过来
        self.X = self.X.append(pd.DataFrame(Xendpic_flatten))
        return
    
def varkkk(arrx,arry):
    s=arrx.shape
    h=s[0]
    w=s[1]
    result=0
    for y in range(h):
        for x in range(w):
            result=result+(int(arrx[x][y])-int(arry[x][y]))**2
                
    return(result)
            
    
    

if __name__ == '__main__':
    N = 1 #生成数据大小
    size_x=64 #图片的宽
    size_y=64 #图片的长
    p_max=9/64/64 #图片中滴点的位置
    
    x_max = 255 #图片中最大值
    x_th = 179 #能看到图案时的threshold
    alpha0 = 0.05 #公式中的alpha，代表自身增长率
    beta0=0.001#公式中的beta，代表周围对齐影响
    step_end = 28 #停止步骤，第150步停止；
    

    
   
        
    data_mem = Data_mem(size_x=size_x,size_y=size_y) #生成存储数据的dataframe
    for i in range(0,N):
        X = load_picfromexcel("data_C.xlsx")
        pic = Generate_pic(x_max=x_max,x_th=x_th,alpha0=alpha0,beta0=beta0,step_end =step_end,X = X)
        plt.figure(1)
        plt.imshow(pic.X0)
        plt.figure(2)
        plt.imshow(pic.Xendpic)
        data_mem.add_sample(pic.X0,pic.Xendpic)
        print(i)
        print(pic)
        Y=load_picfromexcel("data_L.xlsx")
    
    """ for beta1000 in range(10,11):
            beta0=beta1000/10000
            pic = Generate_pic(x_max=x_max,x_th=x_th,alpha0=alpha0,beta0=beta0,step_end =step_end,X = X)
            qaq=pic.Xendpic
            plt.imshow(pic.Xendpic)
            Z=varkkk(qaq,Y)
            print(Z)
            a=list()
            b=list()
            b.append(Z)
            a.append(beta0)
        print(b)
        print(min(b))"""
    writedata_toexcel(data_mem.X,"data_X.xlsx",data_mem.Y,"data_Y.xlsx")
    
    
