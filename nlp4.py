import math, re, sys, json, collections, subprocess, string
from subprocess import PIPE

# Code to call provided scripts
def process(args):
    return subprocess.Popen(args, stdin=PIPE, stdout=PIPE)

def call(process, stdin):
    output = process.stdin.write(stdin + '\n\n')
    line = ''
    while 1:
        l = process.stdout.readline()
        if not l.strip(): break
        line += l
    return line

# Used to test assignment on training data--function to create untagged copy of tagged corpus
def untag(infile, outfile):
    i = open(infile, 'r')
    o = open(outfile, 'w')
    for line in i:
        line = str.split(line)
        if len(line) > 0:
            o.write(line[0])
        o.write('\n')
    i.close()
    o.close()

# Return model in dictionary form for O(1) access from model file
def map_model(model):
    file = open(model, 'r')
    v = collections.defaultdict(default_factory)
    for line in file:
        parsed = str.split(line)
        v[parsed[0]] = float(parsed[1])
    file.close()
    return v

# Dictionary with default value 0
def default_factory():
    return 0;

# Dictionary with default value {} (empty dictionary)
def default_factory_dict():
    return {};

# Tag untagged file using pre-trained model v; write to tagged file
def pretrained_tag(v, untagged, tagged):
    enum_server = process(['python', 'tagger_history_generator.py', 'ENUM'])
    history_server = process(['python', 'tagger_decoder.py', 'HISTORY'])
    untagged_file = open(untagged, 'r')
    tagged_file = open(tagged, 'w')
    sentence = ''
    for line in untagged_file:
        if len(line) > 1: # Middle of sentence
            sentence += line
        else: # End of sentence
            sentence = sentence[:-1] # Remove trailing newline
            histories = str.split(call(enum_server, sentence), '\n') # Enumerate possible histories for sentence
            sentence = str.split(sentence)
            scores = calc_features(histories, sentence, v, 1, -1) # Calculate weights for all histories
            tags = str.split(call(history_server, scores), '\n') # Find highest scoring sequence of histories
            for i in range(len(sentence)): # Extract tag from each history and write to file
                parsed = str.split(tags[i])
                tagged_file.write(sentence[i] + ' ' + parsed[2] + '\n')
            tagged_file.write('\n')
            sentence = ''
    tagged_file.close()
    untagged_file.close()

# Return set of features for model
# Input f indicates which features to use: 0-BIGRAM, TAG; 1-BIGRAM, TAG, SUFFIX; 2-BIGRAM, TAG, PREFIX; 3-BIGRAM, TAG, SUFFIX, PREFIX; 4-BIGRAM, TAG, LENGTH; 5-BIGRAM, TAG, CASE; 6-BIGRAM, TAG, CASE, SUFFIX; 7-BIGRAM, TAG, CASE, PREFIX
def features_set(word, tag, features, f, i):
    if f in (-1, 1, 3, 6):
        features.extend(['SUFFIX:'+word[-3:]+':3:'+tag, 'SUFFIX:'+word[-2:]+':2:'+tag, 'SUFFIX:'+word[-1:]+':1:'+tag])
    if f in (-1, 2, 3, 7):
        features.extend(['PREFIX:'+word[:3]+':3:'+tag, 'PREFIX:'+word[:2]+':2:'+tag, 'PREFIX:'+word[:1]+':1:'+tag])
    if f in (-1, 4):
        features.append('LEN:'+str(len(word))+':'+tag)
    # Determine case of word
    if f in (-1, 5, 6, 7):
        lo = string.lowercase
        up = string.uppercase
        num = string.digits
        pun = string.punctuation
        for i in range(len(word)):
            if i == 0:
                if string.find(lo, word[i]) != -1:
                    type = 'LO'
                elif string.find(up, word[i]) != -1:
                    type = 'SEN'
                elif string.find(num, word[i]) != -1:
                    type = 'NUM'
                elif string.find(pun, word[i]) != -1:
                    type = 'PUN'
                else:
                    type = 'MIX'
                    break
            else:
                if string.find(lo+pun, word[i]) == -1 and type in ['LO', 'SEN']:
                    type = 'MIX'
                    break
                elif string.find(up+pun, word[i]) == -1 and type == 'UP':
                    type = 'MIX'
                    break
                elif string.find(pun, word[i]) == -1 and type == 'PUN':
                    if string.find(lo, word[i]) != -1:
                        type = 'LO'
                    elif string.find(up, word[i]) != -1:
                        type = 'UP'
                    elif string.find(num, word[i]) != -1:
                        type = 'NUM'
                elif string.find(num+pun, word[i]) == -1 and type == 'NUM':
                    type = 'MIX'
                    break
        features.append('CASE:'+type+':'+tag)
    return features

# Extract features from histories
# Score is indicator for return value
# If score == 1, calculate weight for each history and return string of histories with weights
# If score == 0, return dictionary of features for sequence of histories
def calc_features(histories, sentence, v, score, f):
    scores = ''
    for history in histories: # Iterate through histories
        parsed = str.split(history)
        if len(parsed) > 0 and parsed[2] != 'STOP':
            pos = int(parsed[0])-1
            word = str.split(sentence[pos])[0] # Extract word corresponding to history from sentence
            tag = parsed[2] # Extract tag from history
            weight = 0
            standard = ['BIGRAM:'+parsed[1]+':'+tag, 'TAG:'+word+':'+tag]
            features = features_set(word, tag, standard, f, pos) # Create feature set
            for feature in features:
                # Calculating weight for each history
                # Add all scores of features all features of history in dictionary to weight
                # If feature weight is zero, it might not be in dictionary, but weight does not change anyway
                # If a feature is not used by a model, it will not be in v, and thus will not affect the score
                if score == 1:
                    if feature in v:
                        weight += v[feature]
                # Calculating features for sequence of histories
                # Increment counts of all features of history
                # Used only to train, not to tag
                else:
                    if feature in v:
                        v[feature] += 1
                    else:
                        v[feature] = 1
            scores += (history + ' ' + str(weight) + '\n')
    if score == 1:
        return scores[:-1] # Return scored histories
    return v # Return dictionary of weights

# Train model from 'train' file and write to 'model' file
def train_model(train, model, f):
    gold_server = process(['python', 'tagger_history_generator.py', 'GOLD'])
    enum_server = process(['python', 'tagger_history_generator.py', 'ENUM'])
    history_server = process(['python', 'tagger_decoder.py', 'HISTORY'])
    v = collections.defaultdict(default_factory) # Model dictionary
    g = {} # Dictionary to hold gold features for each sentence
    gold_tags = {} # Dictionary to hold gold histories for each sentence
    for itr in range(5): # Iterations of training algorithm
        i = 0
        train_file = open(train, 'r')
        sentence = ''
        for line in train_file:
            if len(line) > 1: # Middle of sentence
                sentence += line
            else: # End of sentence
                sentence = sentence[:-1] # Remove trailing newline
                s = str.split(sentence, '\n')
                if itr == 0: # Gold histories do not change so only need to do this step on first iteration
                    gold = call(gold_server, sentence) # Enumerate gold histories of sentence
                    gold_tags[i] = gold
                    g[i] = calc_features(str.split(gold, '\n'), s, {}, 0, f) # Determine gold features of sentence
                histories = str.split(call(enum_server, sentence), '\n') # Enumerate all possible histories of sentence
                scores = calc_features(histories, s, v, 1, f) # Determine weights of all possible histories
                tags = str.split(call(history_server, scores), '\n') # Find highest scoring history sequence
                if tags != gold_tags[i]: # If highest scoring sequence is not gold sequence, update model
                    features = calc_features(tags, s, {}, 0, f) # Determine features of highest scoring history sequence
                    for feature in (g[i]).keys(): # Increment by gold feature vector
                        v[feature] += g[i][feature]
                    for feature in features.keys(): # Decrement by (wrong) best feature vector
                        v[feature] -= features[feature]
                i += 1
                sentence = ''
        train_file.close()
    model_file = open(model, 'w')
    for feature in v.keys(): # Write all weighted features in trained model to file
        model_file.write(feature + ' ' + str(v[feature]) + '\n')
    model_file.close()
    return v # Return trained model in dictionary form

def main(args):
    if args[1] == 'tag':
        pretrained_tag(map_model(args[2]), args[3], args[4])
    elif args[1] == 'train':
        train_model(args[2], args[3], 1)
    elif args[1] == 'untag':
        untag(args[2], args[3])
    elif args[1] == 'p6':
        train_model(args[2], args[3], int(args[4]))

if __name__ == "__main__":
    main(sys.argv)