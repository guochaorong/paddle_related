# -*- coding: utf-8 -*- 
import pickle
import gzip
import numpy as np
import random
#从数据集中载入数据
def load_data():
    file=gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data=pickle.load(file)
    file.close()
    return training_data,validation_data,test_data

def vectorized_label(j):
    #形状为10行1列
    e=np.zeros((10,1))
    e[j]=1.0
    return e

def data_wrapper():
    train,valid,test=load_data()
    train_inputs=[np.reshape(x,(784,1)) for x in train[0]]
    #print train_inputs[0], len(train_inputs[0])
    train_labels=[vectorized_label(x) for x in train[1]]
    #print train_labels[0], len(train_labels[0])
    train_data=zip(train_inputs,train_labels)
    #print len(train_data)

   #测试集
    test_inputs=[np.reshape(x,(784,1)) for x in test[0]]
    test_data=zip(test_inputs,test[1])

    return train_data,test_data

# sigmoid函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# sigmoid函数的导数
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
    #构造函数初始化网络
    def __init__(self,sizes):
        self.numOfLayers=len(sizes)
        self.sizes=sizes
        #随机初始化偏置和权重
        self.biases=[np.random.randn(i,1) for i in sizes[1:]]
        self.weights=[np.random.randn(j,i) for i,j in zip(sizes[:-1],sizes[1:])]
        print "----biases", self.biases
        print "----weights", self.weights

        #随机梯度下降(训练数据,迭代次数,小样本数量,学习率,是否有测试集)
    def SGD(self,training_data,epochs,mini_batch_size,learning_rate,test_data=None):
        if test_data:
            len_test=len(test_data)
        n=len(training_data)  #训练数据大小
        #迭代过程
        for j in range(epochs):
            print "Epoch {0}:".format(j)
            random.shuffle(training_data)
            #mini_batches是列表中放切割之后的列表
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            #每个mini_batch都更新一次,重复完整个数据集
            for mini_batch in mini_batches:
                #存储C对于各个参数的偏导
                #格式和self.biases和self.weights是一模一样的
                nabla_b=[np.zeros(b.shape) for b in self.biases]
                nabla_w=[np.zeros(w.shape) for w in self.weights]
                eta=learning_rate/len(mini_batch)
                #mini_batch中的一个实例调用梯度下降得到各个参数的偏导
                for x,y in mini_batch:
                    #从一个实例得到的梯度
                    delta_nabla_b,delta_nabla_w=self.backprop(x,y)
                    nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
                    nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
                #每一个mini_batch更新一下参数
                self.biases=[b-eta*nb for b,nb in zip(self.biases,nabla_b)]
                self.weights=[w-eta*nw for w,nw in zip(self.weights,nabla_w)]

            if test_data:
                print "{0}/{1}".format(self.evaluate(test_data),len_test)

    def evaluate(self,test_data):
        #print test_data[0]
        print test_data[0][0] # one of a test data: 784*1
        print self.feedforward(test_data[0][0]) # eva data use current w,b: 10*1
        print [(np.argmax(self.feedforward(x)),y) for (x,y) in [test_data[0]]]
        test_result=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        #print test_result
        return sum(int(i==j) for (i,j) in test_result)


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    #反向传播(对于每一个实例)
    def backprop(self,x,y):
        #存储C对于各个参数的偏导
        #格式和self.biases和self.weights是一模一样的
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        #前向过程
        activation=x
        activations=[x] #存储所有的激活值,一层一层的形式
        zs=[]   #存储所有的中间值(weighted sum)
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        #反向过程
        #输出层error
        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        #非输出层
        for l in range(2,self.numOfLayers):
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(zs[-l])
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())

        return nabla_b,nabla_w

train_data, test_data = data_wrapper()
nn = Network([784, 30, 10])
nn.SGD(train_data, 1, 10, 0.01, test_data=test_data)

