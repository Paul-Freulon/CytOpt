import numpy as np
from scipy.special import logsumexp


def cost(X, y):
    """
    Squared euclidean distance between y and the I points of X.
    if y is the jth point of the support of the distribution,
    the result is the jth column of the cost matrix.

    """
    diff = X-y
    return(np.linalg.norm(diff, axis = 1)**2)


def func_f(lbd, eps, X, Y, j, u, G):
    """
    Compute the function f inside the expectation at the point (y_j, u). 
    """
    
    arg1 = (u - cost(X,Y[j]))/eps
    t1 = logsumexp(arg1)
    
    arg2 = -(G.T).dot(u)/lbd
    t2 = logsumexp(arg2)
    
    result = -eps*t1 -lbd*t2
    
    return(result)
    

def grad_f(lbd, eps, X, Y, j, u, G):
    """
    Compute the gradient with respect to u of the function f inside the expectation
    """
    arg1 = (u - cost(X,Y[j]))/eps
    cor1 = np.max(arg1)
    vec1 = np.exp(arg1-cor1)
    t1 = - vec1/np.sum(vec1)

    arg2 = -(G.T).dot(u)/lbd
    cor2 = np.max(arg2)
    vec2 = np.exp(arg2-cor2)
    t2 = G.dot(vec2)/np.sum(vec2)

    return(t1+t2)

def Gam_mat(Lab_source):
    """
    Compute the Gamma matrix that allows to pass from the class proportions to the weight vector
    """
    if Lab_source.min() == 0:
        K = int(Lab_source.max())+1
        I = Lab_source.shape[0]
        Gamma = np.zeros((I,K))

        for k in range(K):
            Gamma[:,k] = 1/np.sum(Lab_source == k) * np.asarray(Lab_source == k, dtype=float)

    
    else:
        K = int(Lab_source.max())
        I = Lab_source.shape[0]
        Gamma = np.zeros((I,K))

        for k in range(K):
            Gamma[:,k] = 1/np.sum(Lab_source == k+1) * np.asarray(Lab_source == k+1, dtype=float)

    return(Gamma)

def stomax(lbd, eps, X, Y, G, n_iter):
    """
    Robbins-Monro algorithm to compute an approximate of the vector u^* solution of the maximization problem
    """
    I = X.shape[0]
    J = Y.shape[0]
    U = np.zeros(I)

    #Step size policy
    gamma = I*eps/1.9
    c = 0.51

    sample = np.random.choice(I, n_iter)

    for n in range(n_iter):
        idx = sample[n]
        grd = grad_f(lbd, eps, X, Y, idx, U, G)
        U = U + gamma/(n+1)**c * grd

    return(U)

def cytopt_minmax(lbd, eps, X, Y, Lab_source, n_iter):
    """
    The full new procedure to estimate the class proportions in the target data set
    """
    G = Gam_mat(Lab_source)
    u_hat = stomax(lbd, eps, X, Y, G, n_iter)

    # computation of the estimate of the class proportions

    h_hat = np.exp(-(G.T).dot(u_hat)/lbd)
    h_hat = h_hat/h_hat.sum()

    return(h_hat)

def cytopt_minmax_monitor(lbd, eps, X, Y, Lab_source, n_iter, h_true, step=0, power=0):
    """
    Robbins-Monro algorithm to compute an approximate of the vector u^* solution of the maximization problem
    At each step, we evaluate the vector h_hat in order to study the convergence of this algorithm.
    """
    
    I = X.shape[0]
    J = Y.shape[0]
    U = np.zeros(I)
    G = Gam_mat(Lab_source)


    #Step size policy
    if step == 0:
        gamma = I*eps/1.9
    else:
        gamma = step
    
    if power == 0:
        c = 0.51
    else:
        c = power

    sample = np.random.choice(J, n_iter)

    #Estimation of the expectation
    W_storage = np.zeros(n_iter)
    W_storage[0] = func_f(lbd, eps, X, Y, sample[0], U, G)

    #Computation of the Kullback
    kullback_storage = np.zeros(n_iter)

    for n in range(1,n_iter):
        idx = sample[n]
        grd = grad_f(lbd, eps, X, Y, idx, U, G)
        U = U + gamma/(n+1)**c * grd

        W_storage[n] = n/(n+1) * W_storage[n-1] + 1/(n+1) * func_f(lbd, eps, X, Y, idx, U, G)
        
        arg = -(G.T).dot(U)/lbd
        M = np.max(arg)
        
        h_hat = np.exp(arg-M)
        h_hat = h_hat/h_hat.sum()

        Kull_current = np.sum(h_hat * np.log(h_hat/h_true))
        kullback_storage[n] = Kull_current
        #print(h_hat)

    return(h_hat, W_storage, kullback_storage)
