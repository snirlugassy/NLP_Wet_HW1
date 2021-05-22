import pickle
import numpy as np
import pandas as pd
from hw1 import DataProcessing, TaggingFeatureGenerator, Features
from viterbi import Viterbi
from time import time

def error_sum(row):
    return row.drop(row.name).sum()


if __name__ == "__main__":
        output_path = "output.wtag"
        train_data_path = "data/train1.wtag"
        current_path = train_data_path
        current_data = DataProcessing(current_path)
        current_data.process()
        current_histories = current_data.histories


        thresholds = Features.default_thresholds(10)
        thresholds[Features.PWORD_TAG] = 20
        thresholds[Features.WORD_TAG] = 20
        thresholds[Features.PREFIX_TAG] = 50
        thresholds[Features.SUFFIX_TAG] = 50


        gen = TaggingFeatureGenerator(thresholds)
        gen.generate_features(current_histories)
        tags = list(current_data.tags)

        test_data_path = "data/test1.wtag"
        test_data = DataProcessing(test_data_path)
        test_data.process()
        test_histories = test_data.histories


        try:
            with open("weights.pkl", 'rb') as weights_file:
                last_run_params = pickle.load(weights_file)
                w_0 = last_run_params[0]
        except FileNotFoundError:
            print("Weights were not found")
            exit(0)
        mis = 0
        true = 0
        samples = len(test_data.data)
        start = time()
        confusion_matrix = pd.DataFrame(index=tags,columns=tags).fillna(0)
        for i in range(samples):
            sentence = [x[0] for x in test_data.data[i]]
            tags_test = [x[1] for x in test_data.data[i]]
            viterbi = Viterbi(tags, gen.transform, sentence, w_0,5)
            #t0 = time()
            predicted_tags = viterbi.run()
            mis_curr = 0
            true_curr = 0
            for tag,tag_2 in zip(tags_test,predicted_tags):
                if tag == tag_2:
                    true_curr += 1
                else:
                    mis_curr+=1
                confusion_matrix.loc[tag,tag_2] += 1
            mis += mis_curr
            true += true_curr
            with open(output_path, 'a') as output_file:
                new_sentece = []
                for word,tag in zip(sentence,predicted_tags):
                    new_sentece.append(word + "_" + tag +" ")
                new_sentece.append("\n")
                output_file.writelines(new_sentece)
        end = time() - start
        confusion_matrix["error"] = confusion_matrix.apply(lambda row:error_sum(row))
        print(confusion_matrix.nlargest(n=10, columns="error").to_string())
        print("Viterbi run time for " + str(samples) +" was " +str(end/60) + " minutes")
        print("Correctly classified samples: " + str(true))
        print("Misclassified samples: " + str(mis))
        print("Accuracy is: " + str(true/(true+mis)))
