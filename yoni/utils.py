#from numba import jit
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
    take = min(4,len(word))+1
    return [word[-i:] for i in range(1,take)]
 #   return [word[-4:], word[-3:], word[-2:], word[-1:]]


def word_prefixes(word):
    take = min(4,len(word))+1
    return [word[:i] for i in range(1,take)]
  #  return [word[:1], word[:2], word[:3], word[:4]]


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


def softmax(weights, history, f, Y, val):
    """

     calculates the probability of history with every possible label times π(pptag,ptag):
     weights: np.array
     history: word history
     f: history to features transformation function
     Y: All labels
     val: π(pptag,ptag)

     """
    normalizer = 0
    dot = dict()
    for tag in Y:
        temp = np.exp(weight_dot_feature_vec(weights,f(history,tag)))
        dot[tag] = temp * val
        normalizer += temp
    dot = {k: v / normalizer for k, v in dot.items()}
    return dot
