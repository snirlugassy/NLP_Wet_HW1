import numpy as np
import pickle

weights_path = "weights.pkl"

try:
    with open(weights_path, 'rb') as weights_file:
            v = pickle.load(weights_file)
            print(len(v))
except FileNotFoundError:
    print("WALLAK")
    v = np.random.random(10)
    with open(weights_path, "wb") as weights_file:
        pickle.dump(v, weights_file)
        print("dumped")
