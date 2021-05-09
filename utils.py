from numba import jit

def is_numeric(x):
    """
    
    Test if x is numeric or not, mainly used for testing if a string contains numeric expression
    
    """
    
    try:
        float(x)
        return True
    except ValueError:
        return False


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