import string
from collections import OrderedDict, Counter, defaultdict
from enum import Enum
import numpy as np
from numpy.random import choice
from utils import is_numeric, weight_dot_feature_vec
import pickle
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime
from utils import word_prefixes, word_suffixes




class DataProcessing:
    def __init__(self, file_path):
        self.data = list()
        self.histories = list()
        self.tags = set("*")
        with open(file_path, "r") as file:
            self.data = [DataProcessing.split_line(line) for line in file.readlines()]
            # for line in file.readlines():
            #     self.data.append(DataProcessing.split_line(line))

    @staticmethod
    def split_line(line):
        line = line.replace("\n", "").split(" ")
        del line[-1]
        return [x.split("_") for x in line]

    def process(self):
        # tuple with the form - (pptag,ppword,ptag,pword,tag,word)
        for sentence in self.data:
            for i in range(len(sentence)):
                loc = None #Todo: decide if None and just first and last or location, full location adds ~800 features
                #Todo but histories post counter go from 98000 to 108000
                word = sentence[i][0]
                tag = sentence[i][1]

                if tag not in self.tags:
                    self.tags.add(tag)

                if i == 0:
                    ptag = pptag = pword = ppword = "*"
                    loc = "First"
                elif i == 1:
                    ppword = pptag = "*"
                    pword = sentence[i - 1][0]
                    ptag = sentence[i - 1][1]
                else:
                    ppword = sentence[i - 2][0]
                    pptag = sentence[i - 2][1]
                    pword = sentence[i - 1][0]
                    ptag = sentence[i - 1][1]
                if i+1 == len(sentence):
                    loc = "Last"
                self.histories.append((pptag, ppword, ptag, pword, tag, word, loc))
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

    FIRST_WORD_TAG = ("is_first_word", True)
    LAST_WORD_TAG = ("is_last_word", True)
    HAS_HYPHEN_TAG = ("contains_hypen_tag", True)
    ALL_CAP = ("all_word_caps", True)
    PWORD_WORD_TAG = ("pword,word tag combinations",True)
    @staticmethod
    def default_thresholds(threshold = 10):
        return {
            Features.WORD_TAG: 20,
            Features.PWORD_TAG: 20,
            Features.SUFFIX_TAG: 50,
            Features.PREFIX_TAG: 50,
            Features.INIT_CAPITAL_TAG: threshold,
            Features.PUNC_TAG: threshold,
            Features.PPUNC_TAG: threshold,
            Features.SHORT_WORD_TAG: threshold,
            Features.NUMERIC_TAG: threshold,
            Features.TRIGRAM_TAG_SEQ: threshold,
            Features.BIGRAM_TAG_SEQ: threshold,
            Features.UNIGRAM_TAG: threshold,

            Features.FIRST_WORD_TAG: threshold,
            Features.LAST_WORD_TAG: threshold,
            Features.HAS_HYPHEN_TAG: threshold,
            Features.ALL_CAP: threshold,
            Features.PWORD_WORD_TAG: threshold
        }


class TaggingFeatureGenerator:
    def __init__(self, thresholds=Features.default_thresholds(), short_word_length=2):
        self.thresholds = thresholds
        self.short_word_length = short_word_length
        self.feature_statistics = None
        self.features = None
        self.feature_dim = None


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
            location = history[6]

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

            # PREFIX_TAG_COUNT
            for prefix in word_prefixes(word):
                if (prefix, tag) not in self.feature_statistics[Features.PREFIX_TAG]:
                    self.feature_statistics[Features.PREFIX_TAG][(prefix, tag)] = 1
                else:
                    self.feature_statistics[Features.PREFIX_TAG][(prefix, tag)] += 1

            # SUFFIX_TAG_COUNT
            for suffix in word_suffixes(word):
                if (suffix, tag) not in self.feature_statistics[Features.SUFFIX_TAG]:
                    self.feature_statistics[Features.SUFFIX_TAG][(suffix, tag)] = 1
                else:
                    self.feature_statistics[Features.SUFFIX_TAG][(suffix, tag)] += 1

            # INIT_CAPITAL_TAG COUNT
            if word[0].isupper() and pword != "*":
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
            if is_numeric(word):
                if tag not in self.feature_statistics[Features.NUMERIC_TAG]:
                    self.feature_statistics[Features.NUMERIC_TAG][tag] = 1
                else:
                    self.feature_statistics[Features.NUMERIC_TAG][tag] += 1

            # FIRST_WORD_TAG
            if location == "First":
                if tag not in self.feature_statistics[Features.FIRST_WORD_TAG]:
                    self.feature_statistics[Features.FIRST_WORD_TAG][tag] = 1
                else:
                    self.feature_statistics[Features.FIRST_WORD_TAG][tag] += 1

            # LAST_WORD_TAG
            if location == "Last":
                if tag not in self.feature_statistics[Features.LAST_WORD_TAG]:
                    self.feature_statistics[Features.LAST_WORD_TAG][tag] = 1
                else:
                    self.feature_statistics[Features.LAST_WORD_TAG][tag] += 1

            # HAS_HYPHEN_TAG
            if "-" in word:
                if tag not in self.feature_statistics[Features.HAS_HYPHEN_TAG]:
                    self.feature_statistics[Features.HAS_HYPHEN_TAG][tag] = 1
                else:
                    self.feature_statistics[Features.HAS_HYPHEN_TAG][tag] += 1

            # ALL_CAP
            if word.isupper():
                if tag not in self.feature_statistics[Features.ALL_CAP]:
                    self.feature_statistics[Features.ALL_CAP][tag] = 1
                else:
                    self.feature_statistics[Features.ALL_CAP][tag] += 1

            #PWORD_WORD,TAG:
            if (pword,word,tag) not in self.feature_statistics[Features.PWORD_WORD_TAG]:
                self.feature_statistics[Features.PWORD_WORD_TAG][(pword,word,tag)] = 1
            else:
                self.feature_statistics[Features.PWORD_WORD_TAG][(pword,word,tag)] += 1


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
                    if self.feature_statistics[feature][key] > self.thresholds[feature]:
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
            location = history[6]

            # WORD_TAG
            if (word, tag) in self.features[Features.WORD_TAG]:
                feature_vec.append(self.features[Features.WORD_TAG][(word, tag)])

            # PWORD_TAG
            if (pword, tag) in self.features[Features.PWORD_TAG]:
                feature_vec.append(self.features[Features.PWORD_TAG][(pword, tag)])

            # SUFFIX_TAG
            for suffix in word_suffixes(word):
                if (suffix, tag) in self.features[Features.SUFFIX_TAG]:
                    feature_vec.append(self.features[Features.SUFFIX_TAG][(suffix, tag)])

            # PREFIX_TAG
            for prefix in word_prefixes(word):
                if (prefix, tag) in self.features[Features.PREFIX_TAG]:
                    feature_vec.append(self.features[Features.PREFIX_TAG][(prefix, tag)])

            # INIT_CAPITAL_TAG
            if word[0].isupper() and pword != "*":
                if tag in self.features[Features.INIT_CAPITAL_TAG]:
                    feature_vec.append(self.features[Features.INIT_CAPITAL_TAG][tag])

            # PUNC_TAG
            if word in string.punctuation:
                if (word, tag) in self.features[Features.PUNC_TAG]:
                    feature_vec.append(self.features[Features.PUNC_TAG][(word, tag)])

            # PPUNC_TAG
            if pword in string.punctuation:
                if (pword, tag) in self.features[Features.PPUNC_TAG]:
                    feature_vec.append(self.features[Features.PPUNC_TAG][(pword, tag)])

            # SHORT_WORD_TAG - TODO: CONSIDER USING IT ACCORDING TO STATS
            word_length = len(word)
            if word_length <= self.short_word_length and word not in string.punctuation:
                if (word_length, tag) in self.features[Features.SHORT_WORD_TAG]:
                    feature_vec.append(self.features[Features.SHORT_WORD_TAG][(word_length, tag)])

            # NUMERIC_TAG
            if is_numeric(word) and tag in self.features[Features.NUMERIC_TAG]:
                feature_vec.append(self.features[Features.NUMERIC_TAG][tag])
            # TRIGRAM_TAG_SEQ
            if (pptag, ptag, tag) in self.features[Features.TRIGRAM_TAG_SEQ]:
                feature_vec.append(self.features[Features.TRIGRAM_TAG_SEQ][(pptag, ptag, tag)])

            # BIGRAM_TAG_SEQ
            if (ptag, tag) in self.features[Features.BIGRAM_TAG_SEQ]:
                feature_vec.append(self.features[Features.BIGRAM_TAG_SEQ][(ptag, tag)])

            # UNIGRAM_TAG
            if tag in self.features[Features.UNIGRAM_TAG]:
                feature_vec.append(self.features[Features.UNIGRAM_TAG][tag])

            # FIRST_WORD_TAG
            if location == "First" and tag in self.features[Features.FIRST_WORD_TAG]:
                feature_vec.append(self.features[Features.FIRST_WORD_TAG][tag])

            # LAST_WORD_TAG
            if location == "Last" and tag in self.features[Features.LAST_WORD_TAG]:
                feature_vec.append(self.features[Features.LAST_WORD_TAG][tag])

            # HAS_HYPHEN_TAG
            if "-" in word and tag in self.features[Features.HAS_HYPHEN_TAG]:
                feature_vec.append(self.features[Features.HAS_HYPHEN_TAG][tag])

            # ALL_CAP
            if word.isupper() and tag in self.features[Features.ALL_CAP]:
                feature_vec.append(self.features[Features.ALL_CAP][tag])

            #PWORD_WORD_TAG
            if (pword,word,tag) in self.features[Features.PWORD_WORD_TAG]:
                feature_vec.append(self.features[Features.PWORD_WORD_TAG][(pword,word, tag)])
            return np.array(feature_vec)
        return None


def history_likelihood(v, h, f, Y, h_count):
    hl = 0
    hl_grad = np.zeros(len(v))
    dot = np.zeros(len(Y))
    feature_vectors = list()
    for j in range(len(Y)):
        feature_vectors.append(f(h, Y[j]))
        dot[j] = weight_dot_feature_vec(v, feature_vectors[j])
        if Y[j] == h[4]:
            hl += h_count * dot[j]
            for feature in feature_vectors[j]:
                hl_grad[feature] += h_count
    dot = np.exp(dot)
    normalizer = np.sum(dot)
    for i in range(len(feature_vectors)):
        for feature in feature_vectors[i]:
            hl_grad[feature] -= h_count * (dot[i] / normalizer)
    hl -= h_count * np.log(normalizer)
    return hl, hl_grad


def likelihood(v, H, f, Y, reg_param):
    grad = np.zeros(len(v))
    L = 0
    histories_counter = Counter(H)
    for h in histories_counter.keys():
        hl, hl_grad = history_likelihood(v, h, f, Y, histories_counter[h])
        L += hl
        grad = np.add(grad, hl_grad)
    L -= 0.5 * reg_param * np.dot(v, v)
    grad -= reg_param * v
    return (-1) * L, (-1) * grad





if __name__ == "__main__":
    with open("log.txt", "a") as log_file:
        log_file.write("Started: " + datetime.now().isoformat() + "\n")

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
