import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def fet_red(temp,lim=0):
    """
    temp:training data set
    lim:limit which you want it to be bigger
    RETURNS:new dataset and boolean array
    """
    tem=np.array(temp)
    t=np.sum(tem[:,1:],axis=0)

    x=tem[:,1:]
    y=tem[:,0]
    x_t=np.array(x.T[t>lim]).T
    y=np.reshape(y,(len(y),1))
    n_data=np.concatenate((y,x_t),axis=1)
    return n_data,t>lim


def fet_tra(test_data,arr):
    """
    test_data:array which is to be tested
    arr:Boolean array from fet_red
    RETURNS:modified x and y containing id's,and modified test data

    """
    test_data=np.array(test_data)
    x=test_data[:,1:]
    y=test_data[:,0]
    x=np.array(x.T[arr]).T
    y_=np.reshape(y,(len(y),1))
    n_test=np.concatenate((y_,x),axis=1)
    return x,y,n_test


def man_grph(model,k,s=0):
    """
    model:list of models
    s:from where you want to start
    """
    m=len(model)
    for i in range(m):
        plt.plot(model[i][k],label=f"model {i+s}")
    plt.legend()
    plt.show()    
def grph(J_hist):
    """
    J_hist:array which you want to plot
    
    """
    plt.plot(J_hist)
    plt.show()

def grph_list(model,k):
    """
    model:list of models

    """
    temp=[]
    m=len(model)
    for i in range(m):
        temp.append(model[i][k])
    plt.plot(temp)
    plt.show()
    
def divide(training_data,r):
    """
    training_data:Data which you want to divide
    r:ratio in which you want to divide
    RETURN:x_train,y_train,x_csv,y_csv,train,csv
    """
    training_data=np.array(training_data)
    np.random.shuffle(training_data)
    m=len(training_data)
    Ro=int(m*r)
    train=training_data[:Ro]
    csv=training_data[Ro:]
    x_train=train[:,1:]
    y_train=train[:,0]
    x_csv=csv[:,1:]
    y_csv=csv[:,0]
    return x_train,y_train,x_csv,y_csv,train,csv

def normz(X):
    """
    X:a array with n features and m examples
    RETURN:x_,mean,std_dev
    """
    
    m=len(X)
    # n=len(X[0])
    # x_=np.zeros((m,n))
    mean=np.mean(X,axis=0)
    # mean=1/m*np.sum(X,axis=0)
    std_dev=np.std(X,axis=0)
    # std_dev=(1/m*np.sum((X-mean)**2,axis=0))**0.5
    x_=(X-mean)/std_dev
    # for i in range (m):
    #     for j in range(n):
    #         x_[i][j]=(X[i][j]-mean[j])/std_dev[j]
    return x_,mean,std_dev

def transform(x,mean,std_dev):
    """
    x:array to be transformed
    mean:mean of earlier data
    std_dev:std_dev of earlier data
    RETURN:x_
    """
    # m=len(x)
    # n=len(x[0])
    x_=(x-mean)/std_dev
    # x_=np.zeros((m,n))
    # for i in range (m):
    #     for j in range (n):
    #         x_[i][j]=(x[i][j]-mean[j])/std_dev[j]
    return x_


def R2(y_pred,y):
    """
    y_pred:predicted values
    y:original values
    RETURN:r_2_score
    """
    y_mean=np.mean(y)
    y1=np.sum(np.square(y-y_pred))
    y2=np.sum(np.square(y-y_mean))
    r_2_score=1-(y1/y2)
    return r_2_score


def right(ans1,y):
    """
    ans1:predictions
    y:answers
    RETURN: value
    """
    return np.sum(y==ans1)


def one_hot_encoding(y,no_of_classes):
    """
    y:label array
    no_of_classes: no. of classes in label array
    RETURN:y_one
    """
    m=len(y)
    y_one=np.zeros((m,no_of_classes))
    for i in range (m):
        y_one[i][y[i]]=1

    return y_one


# A NEURAL NETWORK CLASS 
class NN:

    # relu -> rectified linear unit 
    # this helps introduce non linearity in the model 
    # used because it mitigates the vanishing gradient problem 

    def Relu(self,z):

        return np.maximum(z,0)
    
    # this is used to get small gradients because relu would have resulted in dead neuroms 
    def leaky_relu(self,z):
        """
        z:array 
        """
        return np.maximum(z,0.01*z)
    
    def leaky_Relu_p(self,z):
        z=np.array(z>0,dtype=np.float32)
        z[z<=0]=0.01
        return z

    def Relu_p(self,z):
        return np.array(z>0,dtype=np.float32)

    #  Converts the input logits into probabilities by exponentiating each element and normalizing by the sum of all exponentials,this results in 1
    def softmax(self,x):
        
        e_x = np.exp(x-np.max(x))
        return (e_x / np.sum(e_x,axis=0))


    

    #categorial cost entropy
    def cat_cost_funct(self,a,y):
        """
        y:array of desired output
        a:array of activations
        RETURN:J
        """
        #computes the loss of for multi class classification 
        J=-np.sum(y*np.log(a))
        return J/len(y)

    #binary cost functions 
    def bi_cost_funct(self,a,y):
        """
        y:array of desired output
        a:array of activations
        RETURN:J
        """
        #computes the loss for binary classification 
        del_=np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))
        J=np.sum(del_)
        return J/len(y)

    #calculate sigmoid ->now z ranges from 0 to 1
    def sgmd(self,z):
        """
        z:array of w*x+b
        """
        return 1.0/(1.0+np.exp(-z))
    
    # find der
    def sgmd_p(self,z):
        """
        z:array of w*x+b
        """
        return self.sgmd(z)*(1-self.sgmd(z))

    # pred fx -> Performs forward propagation through the network for test data, returning the predicted class and actual labels.

    def pred(self,test_data,biases,weights):
        """
        test data:array of the test data
        biases:contains all the biases of the network
        weights:contains all the weights of the network
        RETURN:y_hat,y
        """
        x=test_data[:,1:len(test_data[0])]
        y=test_data[:,0]
        x=x/255
        x=x.T
        for i in range(len(biases)-1):
            x=self.Relu(np.dot(weights[i],x)+biases[i])
            
        x=self.softmax(np.dot(weights[-1],x)+biases[-1])
        # for w,b in zip(weights,biases):
        #     x=sgmd(np.dot(w,x)+b)
        y_fin=np.argmax(x,axis=0)
        return y_fin,y
    
    # This function is designed to output only predictions 
    def pred_test(self,test_data,biases,weights):
        """
        test data:array of the test data
        biases:contains all the biases of the network
        weights:contains all the weights of the network
        RETURN:y_hat
        """
        test_data=np.array(test_data)
        x=test_data[:,1:]
        
        x=x/255
        x=x.T
        for i in range(len(biases)-1):
            x=self.Relu(np.dot(weights[i],x)+biases[i])
            
        x=self.softmax(np.dot(weights[-1],x)+biases[-1])
        # for w,b in zip(weights,biases):
        #     x=sgmd(np.dot(w,x)+b)
        y_fin=np.argmax(x,axis=0)

        return y_fin
    

    # back_prop->Implements the backpropagation algorithm to compute gradients for weights and biases using chain rule of calculus.

    def back_prop(self,biases,weights,x,y):
        """
        x:array with n features and m examples
        y:output array
        biases:biases of every layer
        weights:weights of every layer
        RETURN:w_grad,b_grad
        """
        b_grad=[np.zeros(b.shape) for b in biases]
        w_grad=[np.zeros(w.shape) for w in weights]
        activation=x
        activations=[x]
        zs=[]
        # print(len(biases))
        for i in range(len(biases)):
            z=np.dot(weights[i],activation)+biases[i]
            # print(z)
            # print(np.shape(z))
            zs.append(z)
            if i<len(biases)-1:
                activation=self.leaky_relu(z)
            else:
                activation=self.softmax(z)
            # print(activation)
            # print(np.shape(activation))
            activations.append(activation)

        
        # for b,w in zip(biases,weights):
        #     z=np.dot(w,activation)+b
        #     # print(z)
        #     zs.append(z)
        #     activation=sgmd(z)
        #     # print(activation)
        #     # print(np.shape(activation))
        #     activations.append(activation)
            
            
        
        delta=(activations[-1]-y)
        delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-1])))
        b_grad[-1]=delta_temp
        w_grad[-1]=np.dot(delta,activations[-2].transpose())
        for i in range (2,len(weights)+1):
            delta=np.dot(weights[-i+1].transpose(),delta)*self.leaky_Relu_p(zs[-i])
            # print(np.exp(-zs[-i]))
            delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-i])))
            # print(sgmd_p(zs[-i]))
            
            # print(delta)
            # print(np.shape(delta))
            b_grad[-i]=delta_temp
            w_grad[-i]=np.dot(delta,activations[-i-1].transpose())
            # print(np.shape(delta),np.shape(activations[-i-1].transpose()))
        # delta=(activations[-1]-y)
        # delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-1])))
        # b_grad[-1]=delta_temp
        # w_grad[-1]=np.dot(delta,activations[-2].transpose())
        # for i in range (2,len(weights)+1):
        #     delta=np.dot(weights[-i+1].transpose(),delta)*sgmd_p(zs[-i])
        #     # print(np.exp(-zs[-i]))
        #     delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-i])))
        #     # print(sgmd_p(zs[-i]))
            
        #     # print(delta)
        #     # print(np.shape(delta))
        #     b_grad[-i]=delta_temp
        #     w_grad[-i]=np.dot(delta,activations[-i-1].transpose())
        #     # print(np.shape(delta),np.shape(activations[-i-1].transpose()))
        
        return w_grad,b_grad       
    
    # up_mini->Updates weights and biases using mini-batch gradient descent.
    def up_mini(self,biases,weights,mini_batch,alpha,lmbd):
        """
        mini_batch:current mini batch from the training dataset
        biases:biases of every layer
        weights:weights of every layer
        alpha:learning rate
        RETURN:weights,biases
        """
        # nabla_b = [np.zeros(b.shape) for b in biases]
        # nabla_w = [np.zeros(w.shape) for w in weights]
        x=mini_batch[:,1:]
        x=x/255
        y=mini_batch[:,0]
        y_one=one_hot_encoding(y,10)
        x=x.T
        y_=y_one.T
        nabla_w,nabla_b = self.back_prop(biases,weights,x,y_)
        # for x,y in zip(x,y_one):
            
        #     x=np.reshape(x,(784,1))
        #     y=np.reshape(y,(10,1))
        #     delta_nabla_w,delta_nabla_b = back_prop(biases,weights,x, y)
        #     # print(delta_nabla_w)
        #     # print("yo")
        #     nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        weights=[w-(alpha/len(mini_batch))*wd-(alpha*(lmbd/len(mini_batch)))*w for w,wd in zip(weights,nabla_w)]
        biases=[b-(alpha/len(mini_batch))*bd for b,bd in zip(biases,nabla_b)]
        return weights,biases
        

    # mgdb -> mini batch gradient descent 
    # Initializes weights and biases, then trains the network over a specified number of epochs with mini-batches. It uses weight initialization techniques like He and Xavier initialization for improved training stability.
    def mgd(self,sizes,training_data,mini_batch_sz,alpha,epochs,lmbd,train_data_csv=False):
        """
        x:array of n features m examples
        y:output array
        mini_batch_sz:size of the mini batches
        epochs:no. of times you want to run the algo
        biases:contains all the biases of network
        weights:contains all the weights of the network
        lmbd:lambda for regularization
        train_data_csv:array for cross validation
        RETURN:weights,biases
        """
        biases=[np.random.rand(y,1) for y in sizes[1:]]

        #HE initialization    
        weights=[np.random.uniform(-np.sqrt(6/x),np.sqrt(6/x),(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]    
        
        
        #xavier initialization
        # weights=[np.random.uniform(-np.sqrt(6/(x+y)),np.sqrt(6/(x+y)),(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        
        
        # biases=[np.random.rand(y,1) for y in sizes[1:]]
        # weights=[np.random.rand(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]
        accu=[]    
        for j in range (epochs):
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_sz] for k in range(0,len(training_data),mini_batch_sz)]
            
            for mini_batch in mini_batches:
                weights,biases=self.up_mini(biases,weights,mini_batch,alpha,lmbd)
            
            print(f"epoch:{j}/{epochs}")
            if np.max(train_data_csv):
                y_fin,y_=self.pred(train_data_csv,biases,weights)
                
                acc=np.sum(y_fin==y_)
                print(f"no. of right pred {acc}/{len(train_data_csv)}")
                accu.append((acc/len(train_data_csv))*100)
            # if epochs%10==0:
            #     x_,y_=pred(training_data,biases,weights)
            #     y_=one_hot_encoding(y_,10)
            #     print(cat_cost_funct(x_,y_.T))
        return weights,biases,accu