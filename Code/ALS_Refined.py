import copy
import dill
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

path = "D:\Study\Python Codes\ALSAlgorithm\Data\Dataset.csv"
items_set = None
reviewer_set = None
train_X, valid_X, test_X = None,None,None
all_df = None
K = [10,20,30,40,50]
lambdaP = [0.01,0.1,1,10]
lambdaQ = [0.01,0.1,1,10]
N = 0
class Instance:
    def __init__(self,lp,lq,k,U,V):
        self.lp = lp
        self.lq = lq
        self.k = k
        self.U = copy.deepcopy(U)
        self.V = copy.deepcopy(V)

def save_object_list(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        for ob in obj:
            dill.dump(ob, output)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        dill.dump(obj, output)

def load_object_list(filename,val):
    Olist = []
    with open(filename, 'rb') as output:  # Overwrites any existing file.
        for i in range(val):
            ob1 = dill.load(output)
            Olist.append(ob1)
    return Olist


def load_object(filename):
    with open(filename, 'rb') as output:  # Overwrites any existing file.
        obj = dill.load(output)
    return obj


def Initialize():
    global items_set,reviewer_set,all_df
    all_df = pd.read_csv(path,encoding='utf-8')
    print(all_df.head())
    items = all_df['itemID'].values
    items_set = list(set(items))
    #print(len(items_set))
    reviewers = all_df['reviewerID'].values
    reviewer_set = list(set(reviewers))
    #print(len(reviewer_set))


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename,"rb")
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def Train_Valid_Test_Divide():
    global reviewer_set,items_set,all_df
    print("Set divide")
    total_item_count = len(items_set)
    total_users_count = len(reviewer_set)
    train_users_count = len(reviewer_set) * 6 //10
    del_val_users_count =  len(reviewer_set) * 2//10
    del_test_users_count = len(reviewer_set) * 2//10
    print(total_users_count,train_users_count,del_val_users_count,del_test_users_count)
    print(total_item_count)
    train_user,valid_user,test_user = [],[],[]
    train_item,valid_item,test_item = [],[],[]
    train_rating,valid_rating,test_rating = [],[],[]
    for j in range(total_item_count):
        item_no = items_set[j]
        #item index is j, append that to item lists
        data = all_df.loc[all_df['itemID'] == item_no]
        data_User = data['reviewerID'].values.tolist()
        data_rating = data['rating'].values.tolist()
        #print(data.head())
        for i1 in range(train_users_count):
            train_user_no = reviewer_set[i1]
            if train_user_no in data_User:
                print('Train')
                idx = data_User.index(train_user_no)
                rating = data_rating[idx]
                train_user.append(i1)
                train_item.append(j)
                train_rating.append(rating)
        for i2 in range(train_users_count,train_users_count+del_val_users_count,1):
            valid_user_no = reviewer_set[i2]
            if valid_user_no in data_User:
                print('Valid')
                idx = data_User.index(valid_user_no)
                rating = data_rating[idx]
                valid_user.append(i2-train_users_count)
                valid_item.append(j)
                valid_rating.append(rating)
        for i2 in range(train_users_count+del_val_users_count,train_users_count+del_val_users_count+del_test_users_count,1):
            test_user_no = reviewer_set[i2]
            if test_user_no in data_User:
                print('Test')
                idx = data_User.index(test_user_no)
                rating = data_rating[idx]
                test_user.append(i2-train_users_count-del_val_users_count)
                test_item.append(j)
                test_rating.append(rating)
    M,N = max(train_user)+1, total_item_count
    train_X = sparse.csr_matrix((train_rating,(train_user,train_item)),shape=(M,N))
    D = np.full(len(train_user), 1)
    omega_train = sparse.csr_matrix((D,(train_user,train_item)),shape=(M,N))
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\train",train_X)
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\trainO",omega_train)
    M, N = max(valid_user) + 1, total_item_count
    valid_X = sparse.csr_matrix((valid_rating,(valid_user,valid_item)),shape=(M,N))
    D = np.full(len(valid_user), 1)
    omega_valid = sparse.csr_matrix((D, (valid_user, valid_item)), shape=(M, N))
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\valid", valid_X)
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\validO", omega_valid)
    M, N = max(test_user) + 1, total_item_count
    test_X = sparse.csr_matrix((test_rating,(test_user,test_item)),shape=(M,N))
    D = np.full(len(test_user), 1)
    omega_test = sparse.csr_matrix((D, (test_user, test_item)), shape=(M, N))
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\test", test_X)
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\testO", omega_test)


def SQ_DIFF(X, Omega, User, Products):
    del_X = X - np.dot(User, Products)
    multiple = sparse.csr_matrix(Omega).multiply(sparse.csr_matrix(del_X))
    double_error = sparse.csr_matrix(multiple).multiply(sparse.csr_matrix(multiple))
    return np.sum(double_error) / np.sum(Omega)


def Create_Matrix():
    print("create matrix")
    train_path = "D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\ratings_train.xlsx"
    valid_path = "D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\ratings_validate.xlsx"
    train_df = pd.read_excel(train_path, sheet_name=0, header= None)
    train_X = np.array(train_df._values)
    valid_df = pd.read_excel(valid_path, sheet_name=0, header= None)
    valid_X = np.array(valid_df._values)
    #print(train_X)
    ITrain,JTrain,RTrain = [],[],[]
    row,col = train_X.shape
    #print(row,col)
    count = 0
    for i in range(row):
        for j in range(col):
            if train_X[i][j] > -1:
                ITrain.append(i)
                JTrain.append(j)
                val = train_X[i][j]
                RTrain.append(val)
                count += 1
    Train_X = sparse.csr_matrix((RTrain, (ITrain, JTrain)), shape=(row, col))
    D = np.full(count, 1)
    Omega_Train = sparse.csr_matrix((D, (ITrain, JTrain)), shape=(row, col))
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\train", Train_X)
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\trainO", Omega_Train)
    ITrain, JTrain, RTrain = [], [], []
    row, col = valid_X.shape
    # print(row,col)
    count = 0
    for i in range(row):
        for j in range(col):
            if train_X[i][j] > -1:
                ITrain.append(i)
                JTrain.append(j)
                val = train_X[i][j]
                RTrain.append(val)
                count += 1
    Valid_X = sparse.csr_matrix((RTrain, (ITrain, JTrain)), shape=(row, col))
    D = np.full(count, 1)
    Omega_Valid = sparse.csr_matrix((D, (ITrain, JTrain)), shape=(row, col))
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\valid", Valid_X)
    save_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\validO", Omega_Valid)


def Train():
    global K,lambdaP,lambdaQ
    print('Train on ALS Algorithm')
    #X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\train.npz")
    #Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\trainO.npz")
    X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\train.npz")
    Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\trainO.npz")
    Models = []
    #print(X)
    #print(Omega)
    the_file =  open('D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\TrainReport.txt', 'a')
    for k in K:
        for lp in lambdaP:
            for lq in lambdaQ:
                row, col = X.shape
                print(row, col)
                user = 1.5 * np.random.rand(row, k)
                User = np.asmatrix(user)
                print(User.shape)
                products = 1.5 * np.random.rand(k, col)
                Products = np.asmatrix(products)
                print(Products.shape)
                U = User
                P = Products
                error = 9999999
                Minima = []
                for iter in range(100):
                    print("Iteration ",iter)
                    new_error = SQ_DIFF(X, Omega, User, Products)
                    if error > new_error:
                        error = new_error
                        U = User
                        P = Products
                        print("Error", error)
                        Minima.append(error)
                        for i, omega_r in enumerate(Omega):
                            print("row ",i)
                            omegar = omega_r.toarray()[0]
                            if (sum(omegar) != 0):
                                inv = np.linalg.inv(
                                    np.dot(Products, np.dot(np.diag(omegar), Products.T)) + lp * np.eye(k))
                                Inv = inv.astype(float)
                                x = X[i].toarray()[0]
                                ext = np.dot(Products, np.dot(np.diag(omegar), x.T))
                                Ext = ext.astype(float)
                                val = np.dot(Inv, Ext.T)
                                User[i] = val.T
                                # print("User",User[i])
                        for j, omega_c in enumerate(Omega.T):
                            print("col ", j)
                            omegac = omega_c.toarray()[0]
                            if (sum(omegac) != 0):
                                inv = np.linalg.inv(np.dot(User.T, np.dot(np.diag(omegac), User)) + lq * np.eye(k))
                                Inv = inv.astype(float)
                                x = X[:, j].toarray()
                                ext = np.dot(User.T, np.dot(np.diag(omegac), x))
                                Ext = ext.astype(float)
                                val = np.dot(Inv, Ext)
                                Products[:, j] = val
                User = U
                Products = P
                print("User : ", User)
                print("Product : ", Products)
                Models.append(Instance(lp,lq,k,User,Products))
                strings = "K : "+ str(k) + " Lp : " + str(lp) + " Lq : " + str(lq) + " Error : " + str(error) + "\n"
                the_file.write(strings)
    save_object_list(Models,"D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\trainedModel.txt")
    global N
    N = len(Models)
    the_file.close()
def Validation():
    #global N
    #print("Validate")
    #X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\valid.npz")
    #Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\validO.npz")
    X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\valid.npz")
    Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\validO.npz")
    #print(X)
    #print(Omega)
    Models = load_object_list("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\trainedModel.txt",80)
    error_list = []
    the_file = open('D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\ValidReport.txt', 'a')
    for m in Models:
        products = m.V
        Products = np.asmatrix(products)
        print(Products.shape)
        k = m.k
        lp = m.lp
        lq = m.lq
        #print(k, lp, lq)
        row, col = X.shape
        print(row, col)
        user = np.random.rand(row, k)
        User = np.asmatrix(user)
        print(User.shape)
        for i, omega_r in enumerate(Omega):
            print("row ", i)
            omegar = omega_r.toarray()[0]
            if (sum(omegar) != 0):
                inv = np.linalg.inv(
                    np.dot(Products, np.dot(np.diag(omegar), Products.T)) + lp * np.eye(k))
                Inv = inv.astype(float)
                x = X[i].toarray()[0]
                ext = np.dot(Products, np.dot(np.diag(omegar), x.T))
                Ext = ext.astype(float)
                val = np.dot(Inv, Ext.T)
                User[i] = val.T
                # print("User",User[i])
        new_error = SQ_DIFF(X, Omega, User, Products)
        error_list.append(new_error)
        strings = "K : " + str(k) + " Lp : " + str(lp) + " Lq : " + str(lq) + " Error : " + str(new_error) + "\n"
        the_file.write(strings)
    print(error_list)
    index_min = np.argmin(error_list)
    print(index_min)
    final_model = Models[int(index_min)]
    save_object(final_model,"D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\validatedModel.txt")
    print(final_model.k,final_model.lp,final_model.lq)
    the_file.close()


def Test_All():
    Models = load_object_list("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\trainedModel.txt", 80)
    X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\test.npz")
    Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\testO.npz")
    model = load_object("D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\validatedModel.txt")
    error_list = []
    the_file = open('D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\TestReport.txt', 'a')
    for m in Models:
        products = m.V
        Products = np.asmatrix(products)
        print(Products.shape)
        k = m.k
        lp = m.lp
        lq = m.lq
        # print(k, lp, lq)
        row, col = X.shape
        print(row, col)
        user = np.random.rand(row, k)
        User = np.asmatrix(user)
        print(User.shape)
        for i, omega_r in enumerate(Omega):
            print("row ", i)
            omegar = omega_r.toarray()[0]
            if (sum(omegar) != 0):
                inv = np.linalg.inv(
                    np.dot(Products, np.dot(np.diag(omegar), Products.T)) + lp * np.eye(k))
                Inv = inv.astype(float)
                x = X[i].toarray()[0]
                ext = np.dot(Products, np.dot(np.diag(omegar), x.T))
                Ext = ext.astype(float)
                val = np.dot(Inv, Ext.T)
                User[i] = val.T
                # print("User",User[i])
        new_error = SQ_DIFF(X, Omega, User, Products)
        error_list.append(new_error)
        strings = "K : " + str(k) + " Lp : " + str(lp) + " Lq : " + str(lq) + " Error : " + str(new_error) + "\n"
        the_file.write(strings)
    print(error_list)
    index_min = np.argmin(error_list)
    print(index_min)
    final_model = Models[int(index_min)]
    save_object(final_model, "D:\Study\Python Codes\ALSAlgorithm\Data\XLS\\TestedModel.txt")
    print(final_model.k, final_model.lp, final_model.lq)
    the_file.close()

def Test():
    print("Test")
    X = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\test.npz")
    Omega = load_sparse_csr("D:\Study\Python Codes\ALSAlgorithm\Data\\testO.npz")
    model = load_object("D:\Study\Python Codes\ALSAlgorithm\Data\\validatedModel.txt")
    products = model.V
    Products = np.asmatrix(products)
    print(Products.shape)
    k = model.k
    lp = model.lp
    lq = model.lq
    # print(k, lp, lq)
    row, col = X.shape
    print(row, col)
    user = np.random.rand(row, k)
    User = np.asmatrix(user)
    print(User.shape)
    for i, omega_r in enumerate(Omega):
        print("row ", i)
        omegar = omega_r.toarray()[0]
        if (sum(omegar) != 0):
            inv = np.linalg.inv(
                np.dot(Products, np.dot(np.diag(omegar), Products.T)) + lp * np.eye(k))
            Inv = inv.astype(float)
            x = X[i].toarray()[0]
            ext = np.dot(Products, np.dot(np.diag(omegar), x.T))
            Ext = ext.astype(float)
            val = np.dot(Inv, Ext.T)
            User[i] = val.T
            # print("User",User[i])
    new_error = SQ_DIFF(X, Omega, User, Products)
    print("Error ",new_error)
    neoX = np.dot(User,Products)
    oldX = X.todense()
    print("NeoX ",neoX)
    print("OldX ",oldX)
if __name__ == '__main__':
    #1 Preprocessing Phase
    #Initialize()
    #Train_Valid_Test_Divide()
    Create_Matrix()
    #2 Training Phase
    Train()
    #3  Validation Phase
    #Validation()
    #4 Testing Phase
    #Test()
    #Test_All()
