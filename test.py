import pickle
import numpy as np
from hw1 import DataProcessing, TaggingFeatureGenerator, likelihood, Viterbi
from utils import weight_dot_feature_vec
from collections import Counter
from time import time

if __name__ == "__main__":
    train_data_path = "data/train1.wtag"
    train_data = DataProcessing(train_data_path)
    train_data.process()
    train_histories = train_data.histories
    gen = TaggingFeatureGenerator(threshold=10)
    gen.generate_features(train_histories)
    tags = list(train_data.tags)
    
    # test_data_path = "data/test1.wtag"
    # test_data = DataProcessing(test_data_path)
    # test_data.process()
    # test_histories = test_data.histories

    
    try:
        with open("weights.pkl", 'rb') as weights_file:
            last_run_params = pickle.load(weights_file)
            w_0 = last_run_params[0]
    except FileNotFoundError:
        print("Weights were not found")
        exit(0)
    

    sentence = [x[0] for x in train_data.data[2]]
    viterbi = Viterbi(tags, gen.transform, sentence, w_0)
    t0 = time()
    predicted_tags = viterbi.run()
    print(time()-t0, "seconds passed")
    print(sentence)
    print(predicted_tags)
    
    # L, grad = likelihood(w_0, train_histories, gen.transform, tags)
    # print(L)
    # print(np.linalg.norm(grad))

    # for hist in test_histories:
    #     print(hist)
    #     print(weight_dot_feature_vec(w_0, gen.transform(hist, hist[4])))
        
    