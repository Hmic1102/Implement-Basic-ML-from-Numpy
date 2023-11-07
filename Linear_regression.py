import numpy as np


class LinearReg(object):
    


    def __init__(self, indim=1, outdim=1):
        self.weighted_vector=[]
        pass
    
    def fit(self, X, T):
        n=len(X)
        extend=np.transpose([[1]*n])
        extended=np.hstack((extend,X))
        Xt=np.transpose(extended)
        XtX=np.dot(Xt,extended)
        XtT=np.dot(Xt,T)
        self.weighted_vector= np.linalg.solve(XtX,XtT)
        return self
        

    def predict(self, X):
        n=len(X)
        extend=np.transpose([[1]*n])
        extended=np.hstack((extend,X))
        predicted=np.dot(extended,self.weighted_vector)
        return predicted 


def second_order_basis(x):
    sec_basis=[]
    nrow=len(x)
    ncol=len(x[0])
    for i in range(nrow):
        array=[]
        for j in range(ncol):
            for k in range(j,ncol):
                array.append(x[i][j]*x[i][k])
        sec_basis.append(array)

    return sec_basis
   
