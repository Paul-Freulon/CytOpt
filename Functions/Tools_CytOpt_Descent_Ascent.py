import numpy as np
from scipy.special import logsumexp


def cost(X,y):
    """
    Squared euclidean distance between y and the I points of X.
    if y is the jth point of the support of the distribution,
    the result is the jth column of the cost matrix.

    """
    diff = X-y
    return(np.linalg.norm(diff, axis = 1)**2)


def c_transform(f, X, Y, j, beta, eps=0.1):
    """Calculate the c_transform of f in y_j"""
    arg = (f - cost(X, Y[j]))/eps
    return(eps*( np.log(beta[j])-logsumexp(arg)))


def grad_heps(f, X, y, alpha, eps=0.1):
    """
    Calculate the gradient of h_eps at (y,f)
    """
    arg = (f-cost(X,y))/eps
    a = np.max(arg)
    pi = np.exp(arg-a)
    pi = pi/pi.sum()
    return(alpha-pi)


def h_eps(f, X, Y, j, alpha, beta, eps=0.1):
    """
    Calculate the function h_eps at (y_j, f) whose expectation we want to maximize.
    """
    return(np.sum(f*alpha)+c_transform(f,X,Y,j,beta,eps)-eps)



def Robbins_Monro_Algo(X, Y, alpha, beta, eps=0.1, n_iter=10000):
    """
    Function that calculate the approximation of the Wasserstein distance between alpha and beta
    thanks to the Robbins-Monro Algorithm. X and Y are the supports of the source and target
    distributions. alpha and beta are the weights of the distributions.
    """
    I = X.shape[0]
    J = Y.shape[0]

    #Definition of the step size policy
    gamma = eps/(1.9 * min(beta))
    c = 0.51

    # Storage of the estimates
    W_hat_storage = np.zeros(n_iter)
    Sigma_hat_storage = np.zeros(n_iter)
    h_eps_storage = np.zeros(n_iter)
    h_eps_square = np.zeros(n_iter)

    # Sampling according to the target distribution.
    sample = np.random.choice(a=np.arange(J), size=n_iter, p=beta)

    # Initialisation of the dual vector f.
    f = np.random.random(I)
    f = f - np.mean(f)

    # First iteration to start the loop.
    W_hat_storage[0] = h_eps(f, X, Y, sample[0], alpha, beta, eps)
    h_eps_storage[0] = h_eps(f, X, Y, sample[0], alpha, beta, eps)
    h_eps_square[0] = h_eps(f, X, Y, sample[0], alpha, beta, eps)**2

    #Robbins-Monro Algorithm.

    for k in range(1,n_iter):

        # Sample from the target measure.
        j = sample[k]

        # Update of f.
        f = f + gamma/((k+1)**c) * grad_heps(f, X, Y[j,:], alpha, eps)
        h_eps_storage[k] = h_eps(f, X, Y, j, alpha, beta, eps)

        # Update of the estimator of the regularized Wasserstein distance.
        W_hat_storage[k] = k/(k+1) * W_hat_storage[k-1] + 1/(k+1) * h_eps_storage[k]

       
        # Update of the estimator of the asymptotic variance
        h_eps_square[k] = k/(k+1) * h_eps_square[k-1] + 1/(k+1) * h_eps_storage[k]**2
        Sigma_hat_storage[k] = h_eps_square[k] - W_hat_storage[k]**2

    L = [f, W_hat_storage, Sigma_hat_storage]
    return(L)

def Label_Prop_sto(L_source, f, X, Y, alpha, beta, eps):
    """
    Function that calculates a classification on the target data
    thanks to the approximation of the transport plan and the classification of the source data.
    We got the approximation of the transport plan with the stochastic algorithm.
    """
    print(alpha)
    I = X.shape[0]
    J = Y.shape[0]
    N_cl = L_source.shape[0]
    
    # Computation of the c-transform on the target distribution support.
    f_ce_Y = np.zeros(J)
    for j in range(J):
        f_ce_Y[j] = c_transform(f, X, Y, j, beta, eps)

    print('Computation of ctransform done.')

    L_target = np.zeros((N_cl,J))

    for j in range(J):

        cost_y = cost(X, Y[j])
        arg = (f + f_ce_Y[j] - cost_y)/eps
        P_col = np.exp(arg)
        L_target[:,j] = L_source.dot(P_col)

    clustarget = np.argmax(L_target, axis = 0) + 1
    return([L_target, clustarget])


def diff_simplex(h):
    K = len(h)
    Diff = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i==j:
                Diff[i,j] = (np.exp(h[i])*np.sum(np.exp(h)) - np.exp(2*h[i]))/(np.sum(np.exp(h))**2)
            else:
                Diff[i,j] = - np.exp(h[i]+h[j])/(np.sum(np.exp(h))**2)

    return (Diff)

def cytopt_desas(n_it_grad, X, Lab_source, Y, n_it_sto, step_grad=1/1000, eps=1, cont=True):
    """
    Function that estimates the class proportions in the target data set. It solves the minimization problem with a gradient descent method. At each iteration, the gradient of W^{eps}(alpha, beta) is approximated thanks to the Robbins-Monro algorithm.
    """
    # Definition of the operator D that links the class proportions and the weights.
    I=X.shape[0]
    J=Y.shape[0]
    if min(Lab_source) == 0:
        K = int(max(Lab_source))
        D = np.zeros((I,K+1))
        for k in range(K+1):
            D[:,k] = 1/np.sum(Lab_source == k) * np.asarray(Lab_source == k, dtype=float)

        h = np.ones(K+1)

    else:
        K = int(max(Lab_source))
        D = np.zeros((I,K))
        for k in range(K):
            D[:,k] = 1/np.sum(Lab_source == k+1) * np.asarray(Lab_source == k+1, dtype=float)

        h = np.ones(K)

    #Weights of the target distribution
    beta = 1/J * np.ones(J)

    # Descent-Ascent procedure
    for i in range(n_it_grad):
        prop_classes = np.exp(h)
        prop_classes = prop_classes/np.sum(prop_classes)
        Dif = diff_simplex(h)
        alpha_mod = D.dot(prop_classes)
        f_star_hat = Robbins_Monro_Algo(X, Y, alpha_mod, beta, eps=eps,n_iter=n_it_sto)[0]
        h = h - step_grad*((D.dot(Dif)).T).dot(f_star_hat)
        prop_classes_new = np.exp(h)
        prop_classes_new = prop_classes_new/np.sum(prop_classes_new)
        if i%1000 == 0:
            if cont == True:
                print('Iteration ', i)
                print('Curent h_hat')
                print(prop_classes_new)

    return(prop_classes_new)



def cytopt_desas_monitor(n_it_grad, X, Lab_source, Y, n_it_sto, h_true, step_grad=1/1000, eps=1, cont=True):
    """
    Function that estimates the class proportions in the target data set. It solves the JCPOT minimization problem with a gradient descent method. At each iteration, the gradient of W^{eps}(alpha, beta) is approximated thanks to the Robbins-Monro algorithm. In this algorithm we have the possibility to monitor the evolution of the Kullback divergence between the estimated proportions the benckmark proportions.
    """    
    
    # Definition of the operator D that links the class proportions and the weights.
    I=X.shape[0]
    J=Y.shape[0]
    if min(Lab_source) == 0:
        K = int(max(Lab_source))
        D = np.zeros((I,K+1))
        for k in range(K+1):
            D[:,k] = 1/np.sum(Lab_source == k) * np.asarray(Lab_source == k, dtype=float)

        h = np.ones(K+1)

    else:
        K = int(max(Lab_source))
        D = np.zeros((I,K))
        for k in range(K):
            D[:,k] = 1/np.sum(Lab_source == k+1) * np.asarray(Lab_source == k+1, dtype=float)

        h = np.ones(K)

    #Weights of the target distribution
    beta = 1/J * np.ones(J)

    #Storage of the KL divergence at each iteration
    KL_storage = np.zeros(n_it_grad)

    # Descent-Ascent procedure
    for i in range(n_it_grad):
        prop_classes = np.exp(h)
        prop_classes = prop_classes/np.sum(prop_classes)
        Dif = diff_simplex(h)
        alpha_mod = D.dot(prop_classes)
        f_star_hat = Robbins_Monro_Algo(X, Y, alpha_mod, beta, eps=eps,n_iter=n_it_sto)[0]
        h = h - step_grad*((D.dot(Dif)).T).dot(f_star_hat)
        prop_classes_new = np.exp(h)
        prop_classes_new = prop_classes_new/np.sum(prop_classes_new)
        if i%1000 == 0:
            if cont == True:
                print('Iteration ', i)
                print('Curent h_hat')
                print(prop_classes_new)
        Kull_current = np.sum(prop_classes_new * np.log(prop_classes_new/h_true))
        KL_storage[i] = Kull_current


    # Class proportions and Kullback-Leibler divergence
    return(prop_classes_new, KL_storage)
