import string
from collections import OrderedDict
from enum import Enum
import numpy as np
from utils import is_numeric, weight_dot_feature_vec
import pickle
from scipy.optimize import fmin_l_bfgs_b

# training_data = "/content/drive/MyDrive/Technion/Semester 6/NLP/Homework/Wet 1/data/train1.wtag"
training_data = "data/train1.wtag"
weights_path = "weights.pkl"

class DataProcessing:
    def __init__(self, file_path):
        self.data = list()
        self.histories = list()
        self.tags = set()
        with open(training_data, "r") as file:
            for line in file.readlines():
                self.data.append(DataProcessing.split_line(line))

    @staticmethod
    def split_line(line):
        line = line.replace("\n", "").split(" ")
        del line[-1]
        return [x.split("_") for x in line]

    def process(self):
        # tuple with the form - (pptag,ppword,ptag,pword,tag,word)
        for sentence in self.data:
            for i in range(len(sentence)):
                word = sentence[i][0]
                tag = sentence[i][1]
                
                if tag not in self.tags:
                    self.tags.add(tag)
                                    
                if i == 0:
                    ptag = pptag = pword = ppword = "*"
                elif i == 1:
                    ppword = pptag = "*"
                    pword = sentence[i-1][0]
                    ptag = sentence[i-1][1]
                else:
                    ppword = sentence[i-2][0]
                    pptag = sentence[i-2][1]
                    pword = sentence[i-1][0]
                    ptag = sentence[i-1][1]
                self.histories.append(
                    (pptag, ppword, ptag, pword, tag, word))
        return self.histories


class Features(Enum):
    """
    Feature enumration class

    - This class describes all the features which are used in the TaggingFeatureGenerator generation process.
    - The format of each feature is a tuple (string:Feature name, boolean:Filter the feature based on statistics)

    """

    WORD_TAG = ("word_tag", True)
    PWORD_TAG = ("previous_word_tag", True)
    SUFFIX_TAG = ("suffix_tag", True)
    PREFIX_TAG = ("prefix_tag", True)
    INIT_CAPITAL_TAG = ("init_capital_tag", True)
    PUNC_TAG = ("is_punc_tag", True)
    PPUNC_TAG = ("is_prev_punc_tag", True)
    SHORT_WORD_TAG = ("is_short_word_tag", True)
    NUMERIC_TAG = ("is_numeric_tag", True)
    TRIGRAM_TAG_SEQ = ("trigram_tag_sequence", True)
    BIGRAM_TAG_SEQ = ("bigram_tag_sequence", True)
    UNIGRAM_TAG = ("unigram_tag", True)
    

class TaggingFeatureGenerator:
    def __init__(self, threshold=10, short_word_length=2):
        self.threshold = threshold
        self.short_word_length = short_word_length
        self.feature_statistics = None
        self.features = None
        self.feature_dim = None
        
        self.suffixes = ['s', 'eer', 'er', 'ion', 'ity', 'ment', 'ness', 'or', 'sion', 'ship', 'th', 'able', 'ible', 'al', 'ant',
                         'ary', 'ful', 'ic', 'ious', 'ous', 'ive', 'less', 'y', 'ed', 'en', 'er', 'ing', 'ize', 'ise', 'ly', 'ward', 'wise']
        self.prefixes = ['anti', 'de', 'dis', 'en', 'em', 'fore', 'in', 'im', 'il', 'ir', 'inter',
                         'mid', 'mis', 'non', 'over', 'pre', 're', 'semi', 'sub', 'super', 'trans', 'un', 'under']

    def calc_statistics(self, histories):
        self.feature_statistics = dict()

        for feature in Features:
            self.feature_statistics[feature] = OrderedDict()

        for history in histories:
            pptag = history[0]
            ppword = history[1]
            ptag = history[2]
            pword = history[3]
            tag = history[4]
            word = history[5]

            # WORD_TAG COUNT
            if (word, tag) not in self.feature_statistics[Features.WORD_TAG]:
                self.feature_statistics[Features.WORD_TAG][(word, tag)] = 1
            else:
                self.feature_statistics[Features.WORD_TAG][(word, tag)] += 1
            
            # PWORD_TAG COUNT
            if (pword, tag) not in self.feature_statistics[Features.PWORD_TAG]:
                self.feature_statistics[Features.PWORD_TAG][(pword, tag)] = 1
            else:
                self.feature_statistics[Features.PWORD_TAG][(pword, tag)] += 1

            # PREFIX_TAG COUNT
            for prefix in self.prefixes:
                if word.lower().startswith(prefix):
                    if (prefix, tag) not in self.feature_statistics[Features.PREFIX_TAG]:
                        self.feature_statistics[Features.PREFIX_TAG][(prefix, tag)] = 1
                    else:
                        self.feature_statistics[Features.PREFIX_TAG][(prefix, tag)] += 1

            # SUFFIX_TAG COUNT
            for suffix in self.suffixes:
                if word.lower().endswith(suffix):
                    if (suffix, tag) not in self.feature_statistics[Features.SUFFIX_TAG]:
                        self.feature_statistics[Features.SUFFIX_TAG][(suffix, tag)] = 1
                    else:
                        self.feature_statistics[Features.SUFFIX_TAG][(suffix, tag)] += 1

            # INIT_CAPITAL_TAG COUNT
            if word[0].isupper() and ppword != "*" and pword != "*":
                if tag not in self.feature_statistics[Features.INIT_CAPITAL_TAG]:
                    self.feature_statistics[Features.INIT_CAPITAL_TAG][tag] = 1
                else:
                    self.feature_statistics[Features.INIT_CAPITAL_TAG][tag] += 1

            # PUNC_TAG
            if word in string.punctuation:
                if (word, tag) not in self.feature_statistics[Features.PUNC_TAG]:
                    self.feature_statistics[Features.PUNC_TAG][(word, tag)] = 1
                else:
                    self.feature_statistics[Features.PUNC_TAG][(word, tag)] += 1

            # PPUNC_TAG
            if pword in string.punctuation:
                if (pword, tag) not in self.feature_statistics[Features.PPUNC_TAG]:
                    self.feature_statistics[Features.PPUNC_TAG][(pword, tag)] = 1
                else:
                    self.feature_statistics[Features.PPUNC_TAG][(pword, tag)] += 1

            # SHORT_WORD_TAG
            word_length = len(word)
            if word_length <= self.short_word_length and word not in string.punctuation:
                if (word_length, tag) not in self.feature_statistics[Features.SHORT_WORD_TAG]:
                    self.feature_statistics[Features.SHORT_WORD_TAG][(word_length, tag)] = 1
                else:
                    self.feature_statistics[Features.SHORT_WORD_TAG][(word_length, tag)] += 1

            # TRIGRAM_TAG_SEQ
            if (pptag, ptag, tag) not in self.feature_statistics[Features.TRIGRAM_TAG_SEQ]:
                self.feature_statistics[Features.TRIGRAM_TAG_SEQ][(pptag, ptag, tag)] = 1
            else:
                self.feature_statistics[Features.TRIGRAM_TAG_SEQ][(pptag, ptag, tag)] += 1

            # BIGRAM_TAG_SEQ
            if (ptag, tag) not in self.feature_statistics[Features.BIGRAM_TAG_SEQ]:
                self.feature_statistics[Features.BIGRAM_TAG_SEQ][(ptag, tag)] = 1
            else:
                self.feature_statistics[Features.BIGRAM_TAG_SEQ][(ptag, tag)] += 1

            # UNIGRAM_TAG
            if tag not in self.feature_statistics[Features.UNIGRAM_TAG]:
                self.feature_statistics[Features.UNIGRAM_TAG][tag] = 1
            else:
                self.feature_statistics[Features.UNIGRAM_TAG][tag] += 1
                
            # NUMERIC_TAG - TODO: maybe don't run statistic and just tag as CD
            # if is_numeric(word.replace(",","")):
            # 	if tag not in numeric_tag_count:
            # 		numeric_tag_count[tag] = 1
            # 	else:
            # 		numeric_tag_count[tag] += 1


    def generate_features(self, histories):
        """
        Based on given history, generate the relevant features and save the index to self.features
        """
        if self.feature_statistics == None:
            self.calc_statistics(histories)

        self.features = dict()

        feature_id = 0

        # Filter features iteratively according to the count statistic and the given threshold
        for feature in Features:
            if feature.value[1]:
                filtered_features = OrderedDict()
                for key in self.feature_statistics[feature].keys():
                    if self.feature_statistics[feature][key] > self.threshold:
                        filtered_features[key] = feature_id
                        feature_id += 1
                self.features[feature] = filtered_features
        
        # The feature vector dimension is the total number of feature we've generated
        self.feature_dim = feature_id
        

    def transform(self, history, tag):
        """
        Transform a given history into a feature vector.
        Due to the fact that feature vectors are sparse, returns a sparse representaion of the vector using np.array
        """
        if self.features != None:
            feature_vec = list()
            pptag = history[0]
            ppword = history[1]
            ptag = history[2]
            pword = history[3]
            word = history[5]

            # WORD_TAG
            if (word, tag) in self.features[Features.WORD_TAG]:
                feature_vec.append(self.features[Features.WORD_TAG][(word, tag)])
            
            # PWORD_TAG
            if (pword, tag) in self.features[Features.PWORD_TAG]:
                feature_vec.append(self.features[Features.PWORD_TAG][(pword, tag)])
            
            # SUFFIX_TAG
            for suffix in self.suffixes:
                if word.lower().endswith(suffix) and (suffix, tag) in self.features[Features.SUFFIX_TAG]:
                    feature_vec.append(self.features[Features.SUFFIX_TAG][(suffix, tag)])
            
            # PREFIX_TAG
            for prefix in self.prefixes:
                if word.lower().startswith(prefix) and (prefix, tag) in self.features[Features.PREFIX_TAG]:
                    feature_vec.append(self.features[Features.PREFIX_TAG][(prefix, tag)])
            
            # INIT_CAPITAL_TAG
            if word[0].isupper() and ppword != "*" and pword != "*":
                if tag in self.features[Features.INIT_CAPITAL_TAG]:
                    feature_vec.append(self.features[Features.INIT_CAPITAL_TAG][tag])
                    
            # PUNC_TAG
            if word in string.punctuation:
                if (word,tag) in self.features[Features.PUNC_TAG]:
                    feature_vec.append(self.features[Features.PUNC_TAG][(word,tag)])
                    
            # PPUNC_TAG
            if pword in string.punctuation:
                if (pword,tag) in self.features[Features.PPUNC_TAG]:
                    feature_vec.append(self.features[Features.PPUNC_TAG][(pword,tag)])
                    
            # SHORT_WORD_TAG - TODO: CONSIDER USING IT ACCORDING TO STATS
            word_length = len(word)
            if word_length <= self.short_word_length and word not in string.punctuation:
                if (word_length,tag) in self.features[Features.SHORT_WORD_TAG]:
                    feature_vec.append(self.features[Features.SHORT_WORD_TAG][(word_length,tag)])
                                
            # NUMERIC_TAG
            
            # TRIGRAM_TAG_SEQ
            if (pptag, ptag, tag) in self.features[Features.TRIGRAM_TAG_SEQ]:
                feature_vec.append(self.features[Features.TRIGRAM_TAG_SEQ][(pptag, ptag, tag)])
                
            # BIGRAM_TAG_SEQ
            if (ptag, tag) in self.features[Features.BIGRAM_TAG_SEQ]:
                feature_vec.append(self.features[Features.BIGRAM_TAG_SEQ][(ptag, tag)])
                
            # UNIGRAM_TAG
            if tag in self.features[Features.UNIGRAM_TAG]:
                feature_vec.append(self.features[Features.UNIGRAM_TAG][tag])
                
            return np.array(feature_vec)
        return None


data = DataProcessing(training_data)
data.process()

histories = data.histories
tags = list(data.tags)

gen = TaggingFeatureGenerator(threshold=10)
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
                w_0 = np.random.random(dim)
except FileNotFoundError:
    print("Weights were not found")
    print("Settings random weights")
    w_0 = np.random.random(dim)
    # with open(weights_path, "wb") as weights_file:
    #     pickle.dump(w_0, weights_file)


def likelihood(v, H, f, Y, reg_param=1):
    grad = np.zeros(len(v))
    L = 0
    for h in H:
        dot = np.zeros(len(Y))
        feature_vectors = list()
        for j in range(len(Y)):
            # f(x,y_j)
            feature_vectors.append(f(h,Y[j]))
            dot[j] = weight_dot_feature_vec(v, feature_vectors[j])
            if Y[j] == h[4]:
                L += dot[j]
                for feature in feature_vectors[j]:
                    grad[feature] += 1
        normalizer = np.sum(np.exp(dot))
        
        for i in range(len(feature_vectors)):
            for feature in feature_vectors[i]:
                grad[feature] -= np.exp(dot[i]) / normalizer
            # feature_vectors[i] = feature_vectors[i] * np.exp(dot[i]) / normalizer
        # grad -= sum(feature_vectors)
        L -= np.log(normalizer)
    L -= 0.5 * reg_param * np.dot(v,v)
    grad -= reg_param * v
    return (-1)*L, (-1)*grad
    

# likelihood(w_0, histories, gen.transform, tags)



# # define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter' 
args = (histories, gen.transform, tags, 2)
optimal_params = fmin_l_bfgs_b(func=likelihood, x0=w_0, args=args, maxiter=100, iprint=10)
weights = optimal_params[0]

# # Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# # IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
# weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl' # i identifies which dataset this is trained on
with open(weights_path, 'wb') as weights_file:
    pickle.dump(optimal_params, weights_file)


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

# for i in range(10):
#     h = histories[i]
#     x = softmax(v, h, gen.transform, tags)
#     print(x)
#     print(sum(x))