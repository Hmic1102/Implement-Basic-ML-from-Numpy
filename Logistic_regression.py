import numpy as np


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig
   
   
    


class LogisticReg(object):
    def __init__(self, indim=1):
            self.bias=0
            self.w=[]
            self.weighted=np.zeros((indim+1,1))
            
    
    def set_param(self, weights, bias):
        self.weighted=np.hstack((bias,weights))
        self.w=weights
        self.bias=bias
        
    
    def get_param(self):
        
        return self.weighted[1:],self.weighted[0]

    def compute_loss(self, X, t):
        extend=np.transpose([[1]*len(X)])
        X=np.hstack((extend,X))
        loss=0
        for i in range(len(t)):
            loss-=np.log(sigmoid(t[i]*np.dot(X[i],self.weighted)))
            

        return loss/len(t)



    def compute_grad(self, X, t):
        extend=np.transpose([[1]*len(X)])
        X=np.hstack((extend,X))
        wtx=np.dot(X,self.weighted)
        gra=0
        for i in range(len(X)):
            gra-=(1-sigmoid(t[i]*wtx[i]))*(X[i]*t[i])
        
        return gra/X.shape[0]


        # X: feature matrix of shape [N, d]
        # grad: shape of [d, 1]
        # NOTE: return the average gradient, NOT the sum.



    def update(self, grad, lr=0.001):
        for i in range(len(self.weighted)):
            self.weighted[i]-=grad[i]*lr


    def fit(self, X, t, lr=0.001, max_iters=1000, eps=1e-7):
        # implement the .fit() using the gradient descent method.
        # args:
        #   X: input feature matrix of shape [N, d]
        #   t: input label of shape [N, ]
        #   lr: learning rate
        #   max_iters: maximum number of iterations
        #   eps: tolerance of the loss difference 
        # TO NOTE: 
        #   extend the input features before fitting to it.
        #   return the weight matrix of shape [indim+1, 1]

        loss = 1e10
        for epoch in range(max_iters):
            # compute the loss 
            new_loss = self.compute_loss(X, t)

            # compute the gradient
            grad = self.compute_grad(X, t)

            # update the weight
            self.update(grad, lr=lr)

            # decide whether to break the loop
            if np.abs(new_loss - loss) < eps:
                return self.weighted


    def predict_prob(self, X):
        extend=np.transpose([[1]*len(X)])
        X=np.hstack((extend,X))
        wtx=np.dot(X,self.weighted)
        p=sigmoid(wtx)
        return p
        # implement the .predict_prob() using the parameters learned by .fit()
        # X: input feature matrix of shape [N, d]
        #   NOTE: make sure you extend the feature matrix first,
        #   the same way as what you did in .fit() method.
        # returns the prediction (likelihood) of shape [N, ]


    def predict(self, X, threshold=0.5):
        p=self.predict_prob(X)
        predict=[]
        for i in p:
            if(i>threshold):
                predict.append(1)
            else:
                predict.append(-1)
        return predict
        # implement the .predict() using the .predict_prob() method
        # X: input feature matrix of shape [N, d]
        # returns the prediction of shape [N, ], where each element is -1 or 1.
        # if the probability p>threshold, we determine t=1, otherwise t=-1
        
        
