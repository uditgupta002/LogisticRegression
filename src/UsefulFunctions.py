'''
Created on Oct 31, 2017

@author: udit.gupta
'''

import math
import numpy as np
import time


#Not used since it cannot perform operations on vectors and we require to perform functions over the entire vectors
def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar number

    Return:
    s -- sigmoid(x)
    """
    
    s = 1 / (1 + math.exp(-x))

    return s

def sigmoid(x):
    """
    
    Compute sigmoid of x (x can be a vector or scalar)
    
    Arguments:
    x -- A scalar, vector or matrix

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s
    
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    s = 1 / (1 + np.exp(x))
    ds = s * (1 - s)
    
    return ds

def image2vector(image):
    
    """
    This function will reshape a given image of type (length, height, depth) to (length*height*depth,1)
    
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2],1))
    
    return v

def normalizeRows(x):
    """
    a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. Y
    """
    
    # Compute x_norm 
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    
    # Divide x by its norm.
    x = x / x_norm

    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.

    This code works for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting and divide each row by
    # its corrosponding row value in x_sum. This is basically an element wise division but with broadcasting/
    s = x_exp / x_sum
    
    return s

def vectorized_product_examples():
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### VECTORIZED DOT PRODUCT OF VECTORS ###
    tic = time.time()
    dot = np.dot(x1,x2)
    toc = time.time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED OUTER PRODUCT ###
    tic = time.time()
    outer = np.outer(x1,x2)
    toc = time.time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED ELEMENTWISE MULTIPLICATION ###
    tic = time.time()
    mul = np.multiply(x1,x2)
    toc = time.time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED GENERAL DOT PRODUCT ###
    W=np.zeros(3,x1.shape[1])
    tic = time.time()
    dot = np.dot(W,x1)
    toc = time.time()
    print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
print(softmax(np.zeros((3,2))))
