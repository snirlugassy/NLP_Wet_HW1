import numpy as np
import pickle
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime
from hw1 import DataProcessing, Features, TaggingFeatureGenerator, likelihood

if __name__ == "__main__":
    with open("log.txt", "a") as log_file:
        log_file.write("Started: " + datetime.now().isoformat() + "\n")

    training_data = "data/train1.wtag"
    weights_path = "m1_weights.pkl"
    data = DataProcessing(training_data)
    data.process()

    histories = data.histories
    tags = list(data.tags)

    thresholds = Features.default_thresholds()

    gen = TaggingFeatureGenerator(thresholds)
    gen.generate_features(histories)

    dim = gen.feature_dim
    print("Dimension = ", dim)

    try:
        with open(weights_path, 'rb') as weights_file:
            last_run_params = pickle.load(weights_file)
            w_0 = last_run_params[0]
            print("Weights were found in file")
            if len(w_0) != dim:
                print("Dimension of weights are incorrect, settings random weights")
                w_0 = np.zeros(dim)
    except FileNotFoundError:
        print("Weights were not found")
        print("Settings random weights")
        w_0 = np.random.random(dim)

    # # define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter'
    args = (histories, gen.transform, tags, 0.3)
    optimal_params = fmin_l_bfgs_b(func=likelihood, x0=w_0, args=args, maxiter=30, iprint=1, maxls=8)
    weights = optimal_params[0]

    with open(weights_path, 'wb') as weights_file:
        pickle.dump(optimal_params, weights_file)

    with open("log.txt", "a") as log_file:
        log_file.write("Finished: " + datetime.now().isoformat() + "\n")
