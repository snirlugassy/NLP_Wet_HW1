import pickle
from hw1 import DataProcessing, TaggingFeatureGenerator, Viterbi, Features
from time import time

if __name__ == "__main__":
    data_path = "data/train1.wtag"
    data = DataProcessing(data_path)
    data.process()
    histories = data.histories
    
    thresholds = Features.default_thresholds()
    thresholds[Features.SUFFIX_TAG] = 100
    thresholds[Features.PREFIX_TAG] = 100
    
    gen = TaggingFeatureGenerator(thresholds)
    gen.generate_features(histories)
    tags = list(data.tags)
    
    try:
        with open("weights_backup.pkl", 'rb') as weights_file:
            last_run_params = pickle.load(weights_file)
            w_0 = last_run_params[0]
    except FileNotFoundError:
        print("Weights were not found")
        exit(0)
    

    sentence = [x[0] for x in data.data[2]]
    viterbi = Viterbi(tags, gen.transform, sentence, w_0)
    t0 = time()
    predicted_tags = viterbi.run()
    print(time()-t0, "seconds passed")
    print(sentence)
    print(predicted_tags)