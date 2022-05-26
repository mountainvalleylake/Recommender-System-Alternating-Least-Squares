import copy

import dill
import numpy as np
from scipy import sparse
X = None
Omega = None #Binary matrix for X (0,1)
User = None
Products = None
#Lambda = [0.01,0.1,1,10]
#K = [1,2,3,4,5]
Lambda = 0.1
K = 3

def Make_Matrix(row,col):
    global K,X,User,Products,Omega
    print('Make matrix')
    np.random.seed(3)
    A = np.random.randint(0,5,size=5)
    print(A)
    B = np.random.randint(5,10,size=5)
    print(B)
    C = np.random.randint(1,6,size=5)
    print(C)
    D = [1,1,1,1,1]
    X = sparse.csr_matrix((C,(A,B)),shape=(row,col))
    print(X)
    #X.astype(np.float32)
    Omega = sparse.csr_matrix((D,(A,B)),shape=(row,col))
    #print(Omega)
    #Omega.astype(np.float32)
    user = np.random.rand(row, K)
    User = np.asmatrix(user)
    #print(np.shape(User))
    #print(User)
    #User.astype(np.float32)
    products = np.random.rand(K, col)
    Products = np.asmatrix(products)
    #Products.astype(np.float32)
    #print(np.shape(Products))
    #print(Products)
    mat = np.dot(User,Products)
    #print(mat)
def SQ_DIFF(X,Omega,User,Products):
    del_X = X - np.dot(User,Products)
    multiple = sparse.csr_matrix(Omega).multiply(sparse.csr_matrix(del_X))
    double_error = sparse.csr_matrix(multiple).multiply(sparse.csr_matrix(multiple))
    return np.sum(double_error)/np.sum(Omega)

def ALS():
    global X,User,Products,Omega,K,Lambda
    Minima = []
    print('ALS')
    error = 9999999
    U = User
    P = Products
    for iter in range(100):
        new_error = SQ_DIFF(X, Omega, User, Products)
        if error > new_error:
            error = new_error
            U = User
            P = Products
            print("Error", error)
            Minima.append(error)
            for i, omega_r in enumerate(Omega):
                omegar = omega_r.toarray()[0]
                if(sum(omegar)!= 0):
                    inv = np.linalg.inv(np.dot(Products, np.dot(np.diag(omegar), Products.T))+Lambda * np.eye(K))
                    Inv = inv.astype(float)
                    x = X[i].toarray()[0]
                    ext = np.dot(Products, np.dot(np.diag(omegar), x.T))
                    Ext = ext.astype(float)
                    val = np.dot(Inv, Ext.T)
                    User[i] = val.T
                    #print("User",User[i])
            for j, omega_c in enumerate(Omega.T):
                omegac = omega_c.toarray()[0]
                if(sum(omegac)!=0):
                    inv = np.linalg.inv(np.dot(User.T, np.dot(np.diag(omegac), User))+ Lambda * np.eye(K))
                    Inv = inv.astype(float)
                    x = X[:, j].toarray()
                    ext = np.dot(User.T, np.dot(np.diag(omegac), x))
                    Ext = ext.astype(float)
                    val = np.dot(Inv,Ext)
                    Products[:, j] = val
    User = U
    Products = P
    #print("User : ",User)
    #print("Product : ",Products)
    mat = np.dot(User,Products)
    print("New X : ",mat)
class Obj:
    def __init__(self,x):
        self.x = copy.deepcopy(x)
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        for ob in obj:
            dill.dump(ob, output)
def load_object(filename,val):
    Olist = []
    with open(filename, 'rb') as output:  # Overwrites any existing file.
        for i in range(val):
            ob1 = dill.load(output)
            Olist.append(ob1)
    return Olist
def Test_Class():
    x = [1,2,3,4]
    olist = []
    obj = Obj(x)
    olist.append(obj)
    save_object(olist, "D:\Study\Python Codes\ALSAlgorithm\Data\\trainedModel.txt")
    lst = load_object( "D:\Study\Python Codes\ALSAlgorithm\Data\\trainedModel.txt",val=1)
    print(lst[0].x)
if __name__ == '__main__':
    Make_Matrix(row=10,col=10)
    ALS()
    #Test_Class()