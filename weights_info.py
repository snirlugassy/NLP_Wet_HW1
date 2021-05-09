import pickle
from hw1 import DataProcessing, TaggingFeatureGenerator, likelihood
from time import time
import numpy as np


if __name__ == "__main__":
    data_path = "data/train1.wtag"
    data = DataProcessing(data_path)
    data.process()
    histories = data.histories
    gen = TaggingFeatureGenerator(threshold=10)
    gen.generate_features(histories)
    tags = list(data.tags)
    
        
    try:
        with open("weights.pkl", 'rb') as weights_file:
            last_run_params = pickle.load(weights_file)
            w_0 = last_run_params[0]
    except FileNotFoundError:
        print("Weights were not found")
        exit(0)
    
    t_0 = time()
    
    L, L_grad = likelihood(w_0, histories, gen.transform, tags, reg_param=2)
    print("Likelihood = ", L)
    print("Likelihood gradient norm = ", np.linalg.norm(L_grad))
    
    w_0_norm = np.linalg.norm(w_0)
    print("Weights norm = ", w_0_norm)
    
    print("Time to compute = ", time() - t_0)