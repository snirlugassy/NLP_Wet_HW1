from utils import softmax


def create_history(sentence, pptag, ptag, tag, i):
    word = sentence[i]
    loc = None
    if i == 0:
        pword = ppword = "*"
        loc = "First"
    elif i == 1:
        ppword = "*"
        pword = sentence[i - 1]
    else:
        ppword = sentence[i - 2]
        pword = sentence[i - 1]
    if i + 1 == len(sentence):
        loc = "Last"
    return (pptag, ppword, ptag, pword, tag, word, loc)


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
                table_prev[(t1, t2)] = 0

        table_prev[("*", "*")] = 1

        for k in range(len(self.sentence)):
            paired_tags_prob = dict()
            for ptag in self.tags:
                for t in self.tags:
                    # Part of the beam search - ignores states with 0 score
                    if table_prev[(t, ptag)] > 0:
                        h = create_history(self.sentence, t, ptag, None, k)
                        paired_tags_prob[(t, ptag)] = softmax(self.w, h, self.f, self.tags, table_prev[(t, ptag)])
            for tag in self.tags:
                for ptag in self.tags:
                    prob = []
                    for pptag in self.tags:
                        if table_prev[(pptag, ptag)] > 0:
                            prob.append((pptag, paired_tags_prob[(pptag, ptag)][tag]))
                    if len(prob) != 0:
                        selected = max(prob, key=lambda x: x[1])
                        table_curr[(ptag, tag)] = selected[1]
                        backpointers[(k, ptag, tag)] = selected[0]
                    else:
                        table_curr[(ptag, tag)] = 0

            # Beam search filter goes here
            values = [(key, value) for key, value in table_curr.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            values = [values[i] if i < self.bw else (values[i][0], 0) for i in range(len(values))]
            table_prev = dict(values)
            table_curr = dict()

        tags_predict = list()
        max_ptag, max_tag = max(table_prev.items(), key=lambda x: x[1])[0]
        tags_predict.append(max_tag)
        tags_predict.append(max_ptag)

        for k in range(len(self.sentence) - 3, -1, -1):
            tags_predict.append(backpointers[(k + 2, tags_predict[-1], tags_predict[-2])])
        tags_predict.reverse()
        return tags_predict
