from numba import jit
import numpy as np


def is_numeric(x):
    """
    
    Test if x is numeric or not, mainly used for testing if a string contains numeric expression
    
    """
    
    try:
        float(x)
        return True
    except ValueError:
        return False


def word_suffixes(word):
    return [word[-4:], word[-3:], word[-2:], word[-1:]]


def word_prefixes(word):
    return [word[:1], word[:2], word[:3], word[:4]]


def weight_dot_feature_vec(v,f):
    """
    
    Calculate the dot product of the weights vector v, and the binary feature vector f
    v: np.array not sparse nor compressed
    f: np.array sparse and compresed, if t in f then f_i = 1 and f_i = 0 otherwise
    
    """
    
    product = 0
    for x in f:
        product += v[x]
    return product


# def softmax(weights, history, f, Y):
#     y = Y[0]
#     x = np.zeros(len(Y))
#     normalizer = 0
#     for i in range(len(Y)):
#         y = Y[i]
#         dot = weight_dot_feature_vec(v, f(history,y))
#         x[i] = np.exp(dot)
#         normalizer += x[i]
    
#     return x / normalizer
