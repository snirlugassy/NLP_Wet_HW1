# -*- coding: utf-8 -*-
"""Copy of NLP_1_POS_Tagger.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1266kK5cJ53mYZ4Db4ovbstz4G_lpkHHf

<a href="https://colab.research.google.com/github/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks/blob/master/NLP_1_POS_Tagger.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# <img src="https://img.icons8.com/dusk/64/000000/mind-map.png" style="height:50px;display:inline"> IE 097215 - Technion - Natural Language Processing

## Part 0 - Project Structure
Part Of Speech (POS) tagger is a well known NLP task. As a result, many solutions were proposed to this setup. We present a general solution guidelines to this task (while this is definately not obligatory to use these guidelines to solve HW1). \
A POS tagger can be divided to stages:


*   Pre-training:
    1.   Preprocessing data
    2.   Features engineering  
    3.   Define the objective of the model


*   During training:
    1.   Represent data as feature vectors (Token2Vector) 
    2.   Optimization - We need to tune the weights of the model inorder to solve the objective

*   Inference:
    1.   Use dynamic programing (Viterbi) to tag new data based on MEMM
"""

# Anaconda Environment Setup for HW1 (on Azure machine or on your laptop)

conda env create --file /datashare/hw1/nlp_hw1_env.yml
conda activate nlp_hw1_env

"""## Part 1 - Defining and Extracting Features
In class we saw the importance of extracting good features for NLP task. A good feature is such that (1) appear many times in the data and (2) gives information that is relevant for the label.

### Counting feature appearances in data
We would like to include features that appear many times in the data. Hence we first count the number of appearances for each feature. \
This is done at pre-training step.
"""

from collections import OrderedDict
import string

class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.suffix_prefix_count = OrderedDict()
        self.numbers = OrderedDict()
        self.capitals = OrderedDict()
        self.start_sen = OrderedDict()
        self.punc = OrderedDict()
        self.word_len = OrderedDict()
        # ---Add more count dictionaries here---
        
    def run_all(self, file_path):
      self.get_word_tag_pair_count(file_path)
      self.get_suffix_prefix_count(file_path)
      self.check_numbers(file_path)
      self.check_caps(file_path)
      self.check_punc(file_path)
      self.get_words_length(file_path)

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
          for line in f:
              splited_words = line.split()
              del splited_words[-1]
              for word_idx in range(len(splited_words)):
                  cur_word, cur_tag = splited_words[word_idx].split('_')
                  if (cur_word, cur_tag) not in self.words_tags_count_dict:
                      self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                  else:
                      self.words_tags_count_dict[(cur_word, cur_tag)] += 1
    # --- ADD YOURE CODE BELOW --- #
    
    def get_suffix_prefix_count(self,file_path):
      suffixes = ['s','eer','er','ion','ity','ment','ness','or','sion','ship','th','able','ible','al','ant','ary','ful','ic','ious','ous','ive','less','y','ed','en','er','ing','ize','ise','ly','ward','wise']
      prefixes = ['anti','de','dis','en','em','fore','in','im','il','ir','inter','mid','mis','non','over','pre','re','semi','sub','super','trans','un','under']
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            for s in suffixes:
              if cur_word.lower().endswith(s):
                  if (s + '_s',cur_tag) not in self.suffix_prefix_count:
                      self.suffix_prefix_count[(s + '_s',cur_tag)] = 1
                  else:
                      self.suffix_prefix_count[(s + '_s',cur_tag)] += 1
            for p in prefixes:
              if cur_word.lower().startswith(p):
                  if (p + '_p',cur_tag) not in self.suffix_prefix_count:
                    self.suffix_prefix_count[(p + '_p',cur_tag)] = 1
                  else:
                    self.suffix_prefix_count[(p+ '_p',cur_tag)] += 1

    
    def check_numbers(self,file_path):
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            if cur_word.isdigit:
              if cur_tag not in self.numbers:
                self.numbers[cur_tag] = 1
              else:
                self.numbers[cur_tag] += 1

    def check_caps(self,file_path):
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            if cur_word[0].isupper():
              if word_idx != 0:
                if cur_tag not in self.capitals:
                  self.capitals[cur_tag] = 1
                else:
                  self.capitals[cur_tag] += 1
              else:
                if cur_tag not in self.start_sen:
                  self.start_sen[cur_tag] = 1
                else:
                  self.start_sen[cur_tag] += 1

    def check_punc(self,file_path):
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
            if word_idx != 0 and splited_words[word_idx - 1].split('_')[0] in string.punctuation:
              cur_word, cur_tag = splited_words[word_idx].split('_')
              if (splited_words[word_idx - 1].split('_')[0],cur_tag) not in self.punc:
                self.punc[(splited_words[word_idx - 1].split('_')[0],cur_tag)] = 1
              else: 
                self.punc[(splited_words[word_idx - 1].split('_')[0],cur_tag)] += 1

    def get_words_length(self,file_path):
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
              cur_word, cur_tag = splited_words[word_idx].split('_')
              if (len(cur_word),cur_tag) not in self.word_len:
                self.word_len[(len(cur_word),cur_tag)] = 1
              else: 
                self.word_len[(len(cur_word),cur_tag)] += 1

x = feature_statistics_class()
x.run_all('train1.wtag')
#x.get_word_tag_pair_count('train1.wtag')
#x.get_suffix_prefix_count('train1.wtag')
#x.check_numbers('train1.wtag')
#x.check_caps('train1.wtag')
#x.check_punc('train1.wtag')
#x.get_words_length('train1.wtag')
print(sorted(x.word_len.items(),key = lambda item: item[1],reverse = True))
#print(sorted(x.numbers.items(),key = lambda item: item[1],reverse = True))
#print(sorted(x.capitals.items(),key = lambda item:item[1],reverse = True))
#print(sorted(x.suffix_prefix_count.items(),key = lambda item: item[1],reverse=True))

"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""

class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold                    # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0                     # Total number of features accumulated
        self.n_tag_pairs = 0                          # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = collections.OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
          for line in f:
            splited_words = split(line, (' ', '\n'))
            del splited_words[-1]
            
            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = split(splited_words[word_idx], '_')
                if ((cur_word, cur_tag) not in self.words_tags_dict) \
                        and (self.feature_statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                    self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                    self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs
        
    # --- ADD YOURE CODE BELOW --- #
    def get_is_number(self,file_path):
      with open(file_path) as f:
        for line in f:
          splited_words = line.split()
          del splited_words[-1]
          for word_idx in range(len(splited_words)):
            cur_word, cur_tug = splited_words[word_idx].split('_')
            if cur_word.isdigit:

stat_dict = feature_statistics_class()
stat_dict.

"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.

$$History = (W_{cur}, T_{prev}, T_{next}, T_{cur}, W_{prev}, W_{next}) $$
"""

def represent_input_with_features(history, word_tags_dict):
    """
        Extract feature vector in per a given history
        :param history: touple{word, pptag, ptag, ctag, nword, pword}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    word = history[0]
    pptag = history[1]
    ptag = history[2]
    ctag = history[3]
    nword = history[4]
    pword = history[5]
    features = []

    if (word, ctag) in word_tags_dict:
        features.append(word_tags_dict[(word, ctag)])
        
    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #
    
    return features

"""## Part 2 - Optimization

Recall from tutorial that the log-linear objective is: \
$$(1)\textbf{   } L(v)=\underbrace{\sum_{i=1}^{n} v\cdot f(x_{i},y_{i})}_\text{Linear Term}-\underbrace{\sum_{i=1}^{n}\log(\sum_{y'\in Y} e^{v\cdot f(x_{i},y')})}_\text{Normalization Term} - \underbrace{0.5\cdot\lambda\cdot\|v \|^{2}}_\text{Regularization}$$ 
Where $v$ represents the model weights, $x_{i}$ is the $i'th$ input token, $y_{i}$ is the $i'th$ label and $Y$ represents all possible labels. \
The corresponding gradient is:
$$(2)\textbf{}\frac{\partial L(v)}{\partial v} = \underbrace{\sum_{i=1}^{n} f(x_{i},y_{i})}_\text{Empirical Counts} - \underbrace{\sum_{i=1}^{n}\sum_{y'\in Y} f(x_{i},y') \cdot p(y' | x_{i} ; v)}_\text{Expected Counts} - \underbrace{\lambda\cdot v}_\text{Reg Grad} $$

### How to speed up optimization 
Gradient descent is an iterative optimization method. The log-linear objective presented in equation (1) is a convex problem, which guaranties a convergence for gradient descent iterations (when choosing an appropriate step size). However, in order to converge, many iterations must be performed. Therefore, it is importent to (1) speed up each iteration and to (2) decrease the number of iterations.


#### Decrease the number of iterations 
Notice that by increasing $\lambda$ we can force the algorithm to search for a solution in a smaller search space - which will reduce the number of iterations. However, this is a tredoff, because it will also damage train-set accuracy (Notice that we don't strive to achieve 100% accuracy on the training set, as sometimes by reducing training accuracy we achieve improvement in developement set accuracy).      


#### Decrease iteration duration
Denote the GD update as:
$$ (3) \textbf{  } v_{k+1} = v_{k} + d \cdot \frac{\partial L}{\partial v} $$
where $v_{k}$ is the weight vector at time $k$, $d$ is a constant step size and $L$ is the objective presentend in equation (1).\
In this excersice we are using `fmin_l_bfgs_b`, which is imported from `scipy.optimize`. This is an iterative optimization function which is similar to GD. The function receives 3 arguments:
 

1.   **func** - a function that clculates the objective and its gradient each iteration.
2.   **x0** - initialize values of the model weights.
3.   **args** - the arguments which 'func' is expecting, except for the first argument - which is the model weights.
4.   **maxiter** (optional) - specify a hard limit for number of iterations. (int)
5.   **iprint**  (optional) - specify the print interval (int) 0 < iprint < 99 



Think of ways to efficiently calculate eqautions (1) and (2) according to your features implementation. Furthermore, think which parts must be computed in each iteration, and whether others can be computed once.
"""

def calc_objective_per_iter(w_i, arg_1, arg_2, ...):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i 
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization
        
            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """

    ## Calculate the terms required for the likelihood and gradient calculations
    ## Try implementing it as efficient as possible, as this is repeated for each iteration of optimization.

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1)*likelihood, (-1)*grad

"""Now lets run the code untill we get the optimized weights.

"""

from scipy.optimize import fmin_l_bfgs_b

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pairs(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)
feature2id.get_word_tag_pairs(train_path)

# define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter' 
args = (arg_1, arg_2, ...)
w_0 = np.zeros(n_total_features, dtype=np.float32)
optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]

# Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl' # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

#### In order to load pre-trained weights, just use the next code: ####
#                                                                     #
# with open(weights_path, 'rb') as f:                                 #
#   optimal_params = pickle.load(f)                                   #
# pre_trained_weights = optimal_params[0]                             #
#                                                                     #
#######################################################################

"""## Part 3 - Inference with MEMM-Viterbi
Recall - the MEMM-Viterbi takes the form of:
\begin{align*}
\textbf{Base case: }~~~~~~ &                       \\
                           & \pi(0, *, *)=1        \\
                                                      &                       \\
\textbf{The recursive}     & \textbf{ definition:} \\
                           & \text{For any } k\in\{1,..., n\} ,\text{for any } u \in S_{k-1} \text{ and } v \in S_{k}:\\
                           & ~~~~~~~~~~~~~~~\pi(k, u, v)=\max_{t\in S_{k-2}}{\{\pi(k-1, t, u)\cdot q(v|t, u, w_{[1:n]}, k)\}}\\
\end{align*}
where $S_{k}$ is a set of possible tags at position $k$.



"""

def memm_viterbi():
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    
    return tags_infer

from IPython.display import HTML
from base64 import b64encode
! git clone https://github.com/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks.git

mp4 = open('/content/097215_Natural-Language-Processing_Workshop-Notebooks/ViterbiSimulation.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video width=1000 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

"""Notation:
  - *w* refers to the trained weights vector
  - *u* refers to the previous tag
  - *v* refers to the current tag
  - *t* refers to the tag previous to *u*

The video above presents a vanilla memm viterbi. \
There are several methods to improve the performence of the algorithm, we will specify two of them: 


1.   Dividing the algorithm to multiple processes 
2.   Implementing beam search viterbi, and reducing Beam size 


Notice that the latter might affect the results, hence beam size is required to be chosen wisely.

## Accuracy and Confusion Matrix
![Accuracy and Confusion Matrix](https://raw.githubusercontent.com/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks/master/conf_mat_slide.PNG)

## Interface for creating competition files
In your submission, you must implement a python file named `generate_comp_tagged.py` which generates your tagged competition files for both datasets in a single call.
It should do the following for each dataset:

1. Load trained weights to your model
2. Load competition data file
3. Run inference on competition data file
4. Write results to file according to .wtag format (described in HW1)
5. Validate your results file comply with .wtag format (according to instructions described in HW1)

## <img src="https://img.icons8.com/dusk/64/000000/prize.png" style="height:50px;display:inline"> Credits
* Special thanks to <a href="mailto:taldanielm@campus.technion.ac.il">Tal Daniel</a> , specifically for his viterbi implementation
* By <a href="https://github.com/eyalbd2">Eyal Ben David</a> and <a href="https://github.com/nadavo">Nadav Oved </a>
"""