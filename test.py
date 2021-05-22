import pickle
import numpy as np
import pandas as pd
from hw1 import DataProcessing, TaggingFeatureGenerator, Features
from viterbi import Viterbi
from time import time

def error_sum(row):
    return row.drop(row.name).sum()


if __name__ == "__main__":
        output_path = "m1.wtag"
        train_data_path = "data/train1.wtag"
        current_path = train_data_path
        current_data = DataProcessing(current_path)
        current_data.process()
        current_histories = current_data.histories

        thresholds = Features.default_thresholds(10)

        gen = TaggingFeatureGenerator(thresholds)
        gen.generate_features(current_histories)
        tags = list(current_data.tags)

        test_data_path = "data/test1.wtag"
        test_data = DataProcessing(test_data_path)
        test_data.process()
        test_data_size = len(test_data.data)
        # test_histories = test_data.histories


        try:
            with open("weights.pkl", 'rb') as weights_file:
                last_run_params = pickle.load(weights_file)
                w_0 = last_run_params[0]
        except FileNotFoundError:
            print("Weights were not found")
            exit(0)
            
        predictions = list()
        incorrect_count = 0
        correct_count = 0
        start_time = time()
        incorrect_tags = dict()
        confusion_matrix = pd.DataFrame(index=tags,columns=tags).fillna(0)
        
        
        for i in range(test_data_size):
            # print(i+1,'/',test_data_size)
            sentence = [x[0] for x in test_data.data[i]]
            test_tags = [x[1] for x in test_data.data[i]]
            viterbi = Viterbi(tags, gen.transform, sentence, w_0,5)
            predicted_tags = viterbi.run()
            predictions.append((sentence, predicted_tags))
            
            for t,p in zip(test_tags, predicted_tags):
                if t == p:
                    correct_count += 1
                else:
                    incorrect_count+=1
                    if t in incorrect_tags:
                        incorrect_tags[t] += 1
                    else:
                        incorrect_tags[t] = 1
                confusion_matrix.loc[t,p] += 1

        end_time = time() - start_time
        
        with open(output_path, 'w') as output_file:
            for sentence, predicted_tags in predictions:
                for w,t in zip(sentence, predicted_tags):
                    output_file.write("{}_{} ".format(w,t))
                output_file.write("\n")
        
        # confusion_matrix["error"] = confusion_matrix.apply(lambda row:error_sum(row))
        # print(confusion_matrix.nlargest(n=10, columns="error").to_string())
        
        top_incorrect_tags = sorted(incorrect_tags.items(), key=lambda x:x[1], reverse=True)[:10]
        top_incorrect_tags = [t[0] for t in top_incorrect_tags]
        
        confusion_matrix = confusion_matrix.loc[top_incorrect_tags][top_incorrect_tags]
        
        print("Viterbi run time for ", test_data_size, "sentences was", end_time/60, " minutes")
        print("Correctly classified tags:", correct_count)
        print("Misclassified tags:", incorrect_count)
        print("Accuracy is:", correct_count/(correct_count+incorrect_count))
        
        print("Top 10 incorrect tags confusion matrix:")
        print(confusion_matrix)
