# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from optimizers import *
from math import *
import matplotlib.pyplot as plt
import re
from itertools import *

# Command-line arguments to the system -- you can extend these if you want, but you shouldn't need to modify any of them
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


# Wrapper for an example of the person binary classification task.
# tokens: list of string words
# labels: list of (0, 1) where 0 is non-name, 1 is name
class PersonExample(object):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


# Changes NER-style chunk examples into binary classification examples.
def transform_for_classification(ner_exs):
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)


# Person classifier that takes counts of how often a word was observed to be the positive and negative class
# in training, and classifies as positive any tokens which are observed to be positive more than negative.
# Unknown tokens or ties default to negative.
class CountBasedPersonClassifier(object):
    def __init__(self, pos_counts, neg_counts):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts
        
    def predict(self, tokens, idx):
        if self.pos_counts.get_count(tokens[idx]) > self.neg_counts.get_count(tokens[idx]):
            return 1
        else:
            return 0


# "Trains" the count-based person classifier by collecting counts over the given examples.
def train_count_based_binary_classifier(ner_exs):
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts.increment_count(ex.tokens[idx], 1.0)
            else:
                neg_counts.increment_count(ex.tokens[idx], 1.0)
    return CountBasedPersonClassifier(pos_counts, neg_counts)

# "Real" classifier that takes a weight vector
class PersonClassifier(object):
    def __init__(self, weights, indexer):
        self.weights = weights
        self.indexer = indexer

    # Makes a prediction for token at position idx in the given PersonExample
    def predict(self, tokens, idx):
        is_quotes_present = False
        counts = {}
        for w in tokens:
            if w == '"':
                is_quotes_present = True
            if w in counts:
                counts[w]+=1
            else:
                counts[w]=1
        feature_positions = []
        word = tokens[idx]
        word_length = len(word)
        pat, sum_pat = get_pattern(word) 
        if self.indexer.contains("sum_pat=" + sum_pat):
            feature_positions.append(self.indexer.index_of("sum_pat="+ sum_pat))
        if idx == 0:
            feature_positions.append(self.indexer.index_of("first word"))
        if idx == len(tokens) - 1:
            feature_positions.append(self.indexer.index_of("last word"))
        if idx != 0:
            prev_pat, prev_sum_pat = get_pattern(tokens[idx-1])
            if self.indexer.contains("prev_sum_pat=" + prev_sum_pat):
                feature_positions.append(self.indexer.index_of("prev_sum_pat="+ prev_sum_pat))
            if self.indexer.contains("prev_word="+ tokens[idx-1]):
                feature_positions.append(self.indexer.index_of("prev_word="+ tokens[idx-1]))
            else:
                feature_positions.append(self.indexer.index_of("new prev_word"))
        if idx != len(tokens) - 1:
            next_pat, next_sum_pat = get_pattern(tokens[idx+1])
            if self.indexer.contains("next_sum_pat=" + next_sum_pat):
                feature_positions.append(self.indexer.index_of("next_sum_pat=" + next_sum_pat))
            if self.indexer.contains("next_word="+ tokens[idx+1]):
                feature_positions.append(self.indexer.index_of("next_word="+ tokens[idx+1]))
            else:
                feature_positions.append(self.indexer.index_of("new next_word"))
        if idx != 0 and idx != len(tokens) - 1: 
            if self.indexer.contains("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat):
                feature_positions.append(self.indexer.index_of("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat))
            if self.indexer.contains(self.indexer.index_of("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat)):
                feature_positions.append(self.indexer.index_of("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat))
        if is_quotes_present:
            feature_positions.append(self.indexer.index_of("quotes"))
        if word_length <= 5:
            feature_positions.append(self.indexer.index_of("len<=5"))
        if word_length >= 6 and word_length <=10:
            feature_positions.append(self.indexer.index_of("len [6 to 10]"))
        if word_length >= 11 and word_length <=15:
            feature_positions.append(self.indexer.index_of("len [11 to 15]"))
        if word_length >= 16 and word_length <=20:
            feature_positions.append(self.indexer.index_of("len [16 to 20]"))
        if word_length > 20:
            feature_positions.append(self.indexer.index_of("len>20"))
        if self.indexer.contains("len="+str(word_length)):
            feature_positions.append(self.indexer.index_of("len="+str(word_length)))
        if self.indexer.contains("current word=" + word):
            feature_positions.append(self.indexer.index_of("current word=" + word))
        else:
            feature_positions.append(self.indexer.index_of("new word"))
        if counts[word] == 1:
            feature_positions.append(self.indexer.index_of("word appeared once"))
        if counts[word] == 2:
            feature_positions.append(self.indexer.index_of("word appeared twice"))
        if counts[word] > 2:
            feature_positions.append(self.indexer.index_of("word appeared more than twice"))
        for f in feature_positions:
            if f == -1:
                raise Exception("Feature does not exist!")
        score = score_indexed_features(feature_positions, self.weights)
        e_score = exp(score)
        prob_name = e_score/(1+e_score)
        if prob_name >= 0.5:
            return 1
        else:
            return 0
def train_classifier(ner_exs):
    features = Indexer()
    words = []
    labels = []
    total_exs= sum([len(ex) for ex in ner_exs])
    w = -1 
    feature_positions = {}
    for ex in ner_exs:
        is_quotes_present = False
        counts = {}
        for tok in ex.tokens:
            if tok == '"':
                is_quotes_present = True
            if tok in counts:
                counts[tok]+=1
            else:
                counts[tok]=1
        for i in range(0,len(ex)):
            w += 1
            word = ex.tokens[i]
            words.append(word)
            labels.append(ex.labels[i])
            word_length = len(word)
            features.get_index("new word")
            features.get_index("new prev_word")
            features.get_index("new next_word")
            feature_positions[w] = []
            pat, sum_pat = get_pattern(word) 
            feature_positions[w].append(features.get_index("sum_pat="+sum_pat))
            if i != 0:
                prev_pat, prev_sum_pat = get_pattern(ex.tokens[i-1])
                feature_positions[w].append(features.get_index("prev_sum_pat="+ prev_sum_pat))
                feature_positions[w].append(features.get_index("prev_word="+ ex.tokens[i-1]))
            else:
                feature_positions[w].append(features.get_index("first word"))
            if i != len(ex) - 1:
                next_pat, next_sum_pat = get_pattern(ex.tokens[i+1])
                feature_positions[w].append(features.get_index("next_sum_pat=" + next_sum_pat))
                feature_positions[w].append(features.get_index("next_word="+ ex.tokens[i+1]))
            else:
                feature_positions[w].append(features.get_index("last word"))
            if i != 0 and i != len(ex) - 1: 
                feature_positions[w].append(features.get_index("next_sum_pat=" + next_sum_pat + "prev_sum_pat=" + prev_sum_pat))
                feature_positions[w].append(features.get_index("nsp=" + next_sum_pat + "psp=" + prev_sum_pat + "csp="+sum_pat))
            if is_quotes_present:
                feature_positions[w].append(features.get_index("quotes"))
            if word_length <= 5:
                feature_positions[w].append(features.get_index("len<=5"))
            if word_length >=6 and word_length <= 10:
                feature_positions[w].append(features.get_index("len [6 to 10]"))
            if word_length >= 11 and word_length <=15:
                feature_positions[w].append(features.get_index("len [11 to 15]"))
            if word_length >= 16 and word_length <=20:
                feature_positions[w].append(features.get_index("len [16 to 20]"))
            if word_length > 20:
                feature_positions[w].append(features.get_index("len>20"))
            feature_positions[w].append(features.get_index("len="+str(word_length)))
            feature_positions[w].append(features.get_index("current word=" + word))
            if counts[word] == 1:
                feature_positions[w].append(features.get_index("word appeared once"))
            if counts[word] == 2:
                feature_positions[w].append(features.get_index("word appeared twice"))
            if counts[word] > 2:
                feature_positions[w].append(features.get_index("word appeared more than twice"))
            
    num_features = len(features)
    weights = np.random.randn(num_features) * 0.001
    f1_train = []   
    f1_dev = []   
    alpha = 0.1
    grad_ascent = SGDOptimizer(weights,alpha)
    num_epochs = 58
    for epoch_num in range(num_epochs):
        indices = np.random.choice(total_exs, total_exs)
        indices = np.arange(total_exs)
        for w in indices:
            score = grad_ascent.score(feature_positions[w])
            e_score = exp(score)
            sigmoid = e_score/(1+e_score)
            label_minus_prob = labels[w] - sigmoid #should be close to 0   
            gradient = Counter()
            for feat_pos in feature_positions[w]:
                gradient.set_count(feat_pos, label_minus_prob)
            grad_ascent.apply_gradient_update(gradient,1)
        print(epoch_num)
    return PersonClassifier(grad_ascent.weights, features)     

def get_pattern(word):
    pat = ""
    for c in word:
        if c.isupper():
            pat += "A"
        elif c.islower():
            pat += "a"
        elif c.isdigit():
            pat += "0"
        else:
            pat += c
    summarized_pattern = "".join([k for k, g in groupby(pat)])
    return pat, summarized_pattern

def evaluate_classifier(exs, classifier):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            if prediction == ex.labels[idx]:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if ex.labels[idx] == 1:
                num_gold += 1
            if prediction == 1 and ex.labels[idx] == 1:
                num_pos_correct += 1
            num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec/(prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)
    return f1

# Runs prediction on exs and writes the outputs to outfile, one token per line
def predict_write_output_to_file(exs, classifier, outfile):
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



