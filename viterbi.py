import pickle
from hw1 import DataProcessing, TaggingFeatureGenerator, Features
from utils import softmax
from time import time
import numpy as np

def create_history(sentence, pptag, ptag, tag, i):
    word = sentence[i]
                        
    if i == 0:
        ptag = pptag = pword = ppword = "*"
    elif i == 1:
        ppword = pptag = "*"
        pword = sentence[i-1]
    else:
        ppword = sentence[i-2]
        pword = sentence[i-1]
    return (pptag, ppword, ptag, pword, tag, word, i , len(sentence))

class Viterbi:
    def __init__(self, tags, feature_gen_transform, sentence, weights, beam_width=20):
        self.tags = tags
        self.f = feature_gen_transform
        if isinstance(sentence, str):
            self.sentence = sentence.split(" ")
        else:
            self.sentence = sentence
        self.w = weights
        self.bw = beam_width

   
    def run(self):
        table_prev = dict()
        table_curr = dict()
        backpointers = dict()
        
        for t1 in self.tags:
            for t2 in self.tags:
                table_prev[(t1,t2)] = 0
                # table_curr[(t1,t2)] = 0
        
        table_prev[("*","*")] = 1
        

        for k in range(len(self.sentence)):
            print("Viterbi k =",k)
            # TODO: if k==0 or k==1
            for tag in self.tags:
                for ptag in self.tags:
                    max_p_hist = 0
                    max_t = None
                    for t in self.tags:
                        # Part of the beam search - ignores states with 0 score
                        if table_prev[(t,ptag)] > 0:
                            h = create_history(self.sentence, t, ptag, tag, k)
                            p_hist = softmax(self.w, h, self.f, self.tags) * table_prev[(t,ptag)]
                            if p_hist > max_p_hist:
                                max_p_hist = p_hist
                                max_t = t
                    table_curr[(ptag,tag)] = max_p_hist
                    backpointers[(k,ptag,tag)] = max_t
                    
            # Beam search filter goes here
            values = [(key,value) for key,value in table_curr.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            values = [values[i] if i < self.bw else (values[i][0],0) for i in range(len(values))]
            table_prev = dict(values)
            table_curr = dict()
        
        tags_predict = list()
        max_ptag, max_tag = max(table_prev.items(), key=lambda x:x[1])[0]
        tags_predict.append(max_tag)
        tags_predict.append(max_ptag)
        
        for k in range(len(self.sentence)-3, -1, -1):
            tags_predict.append(backpointers[(k+2, tags_predict[-1], tags_predict[-2])])
        
        return tags_predict
        

if __name__ == "__main__":
    data_path = "data/train1.wtag"
    data = DataProcessing(data_path)
    data.process()
    histories = data.histories
    
    thresholds = Features.default_thresholds()
    # thresholds[Features.SUFFIX_TAG] = 10
    # thresholds[Features.PREFIX_TAG] = 10
    
    gen = TaggingFeatureGenerator(thresholds)
    gen.generate_features(histories)
    tags = list(data.tags)
    
    try:
        with open("weights.pkl", 'rb') as weights_file:
            last_run_params = pickle.load(weights_file)
            w_0 = last_run_params[0]
            if len(w_0) != gen.feature_dim:
                print("Dimension of weights are incorrect, settings random weights")
                w_0 = np.random.random(gen.feature_dim)
    except FileNotFoundError:
        print("Weights were not found")
        exit(0)
    
    test_i = 200
    sentence = [x[0] for x in data.data[test_i]]
    print(sentence)
    ground_truth = [x[1] for x in data.data[test_i]]
    viterbi = Viterbi(tags, gen.transform, sentence, w_0, beam_width=50)
    t0 = time()
    predicted_tags = viterbi.run()
    print(time()-t0, "seconds passed")
    print(sentence)
    print(predicted_tags)
    print(ground_truth)
    print([predicted_tags[i]==ground_truth[i] for i in range(len(ground_truth))])
        
