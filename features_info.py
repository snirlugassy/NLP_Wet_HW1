import pickle
import numpy as np
import pandas as pd
from hw1 import DataProcessing, TaggingFeatureGenerator, Features
from viterbi import Viterbi
from time import time

if __name__ == "__main__":
    training_data = "data/train1.wtag"
    weights_path = "weights.pkl"
    data = DataProcessing(training_data)
    data.process()

    histories = data.histories
    tags = list(data.tags)

    thresholds = Features.default_thresholds()

    gen = TaggingFeatureGenerator(thresholds)
    gen.generate_features(histories)

    dim = gen.feature_dim
    print("Dimension = ", dim)
    
    for feature in gen.features.keys():
        print(feature, len(gen.features[feature]))