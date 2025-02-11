import os
import numpy as np
import scipy.sparse

'''
K=40
os.system("g++ -std=c++17 -O3 -DORD={} -march=native -shared -flto -fPIC -o convert_dist_ad.so convert_dist_ad.cpp -I/opt/homebrew/Cellar/boost/1.87.0/include -L/opt/homebrew/Cellar/boost/1.87.0/lib -lboost_math_tr1 -lm".format(K))
ctypes = np.ctypeslib.ctypes
lib = ctypes.CDLL('./convert_dist_ad.so')
lib.convert_dist.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
'''

def compile_c_code(K=40):
    global lib, ctypes
    os.system(f"g++ -std=c++17 -O3 -DORD={K} -march=native -shared -flto -fPIC -o convert_dist_ad.so convert_dist_ad.cpp -I/opt/homebrew/Cellar/boost/1.87.0/include -L/opt/homebrew/Cellar/boost/1.87.0/lib -lboost_math_tr1 -lm")
    ctypes = np.ctypeslib.ctypes
    lib = ctypes.CDLL('./convert_dist_ad.so')
    lib.convert_dist.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    
def convert(X, K=40):
    ans = np.zeros(K)
    lib.convert_dist(np.ascontiguousarray(X, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ans.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return ans

def return_to_first_return(X, K=40):
    return np.apply_along_axis(convert, 1, X, K=K)

def convert_approx(X):
    F = 1 - 1/np.fft.rfft(np.append(X, np.zeros(len(X))))
    return np.fft.irfft(F)[:len(X)]
    

def first_return_dist(A, K=40, use_fft=False): 
    if not use_fft:
        compile_c_code(K)
    W = A / A.sum(axis=0)
    p = np.eye(A.shape[0])
    X = np.ones((A.shape[0], K))
    
    for t in range(1, K):
        p = W @ p
        X[:, t] = p.diagonal()
        
    if use_fft:
        for i in range(A.shape[0]):
           X[i, :] = convert_approx(X[i, :]) 
        return X
        
    return return_to_first_return(X, K)
    


if __name__=="__main__":
    import networkx as nx
    G = nx.watts_strogatz_graph(100, 4, 0.2)
    A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))
    print(first_return_dist(A))
