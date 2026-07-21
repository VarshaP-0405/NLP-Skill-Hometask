import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

###############################################
# 1. LOAD DATASET
###############################################

df = pd.read_csv("ner_dataset.csv", encoding="latin1")

df = df[['Sentence #','Word','POS']]

# Fill sentence numbers
df['Sentence #'] = df['Sentence #'].ffill()

###############################################
# 2. CREATE SENTENCES
###############################################

sentences = []

for s, group in df.groupby("Sentence #"):
    words = group["Word"].tolist()
    tags = group["POS"].tolist()
    sentences.append((words, tags))

print("Total Sentences:", len(sentences))

###############################################
# 3. TRAIN TEST SPLIT
###############################################

train_sentences, test_sentences = train_test_split(
    sentences,
    test_size=0.2,
    random_state=42
)

###############################################
# 4. BUILD VOCABULARY
###############################################

tags = sorted(list(set(df.POS)))

words = sorted(list(set(df.Word)))

tag2idx = {t:i for i,t in enumerate(tags)}
idx2tag = {i:t for t,i in tag2idx.items()}

word2idx = {w:i for i,w in enumerate(words)}

num_tags = len(tags)

###############################################
# 5. COUNT MATRICES
###############################################

initial_counts = np.ones(num_tags)

transition_counts = np.ones((num_tags, num_tags))

emission_counts = {}

for tag in tags:
    emission_counts[tag] = {}

###############################################
# TRAIN
###############################################

for words_seq, tags_seq in train_sentences:

    initial_counts[tag2idx[tags_seq[0]]] += 1

    for i in range(len(tags_seq)):

        tag = tags_seq[i]
        word = words_seq[i]

        emission_counts[tag][word] = emission_counts[tag].get(word,0)+1

        if i>0:
            prev = tag2idx[tags_seq[i-1]]
            curr = tag2idx[tag]

            transition_counts[prev,curr]+=1

###############################################
# 6. INITIAL PROBABILITY
###############################################

initial_prob = initial_counts/initial_counts.sum()

###############################################
# 7. TRANSITION PROBABILITY
###############################################

transition_prob = transition_counts / transition_counts.sum(axis=1,keepdims=True)

###############################################
# 8. EMISSION PROBABILITY
###############################################

emission_prob = {}

for tag in tags:

    total = sum(emission_counts[tag].values()) + len(words)

    emission_prob[tag] = {}

    for w,c in emission_counts[tag].items():

        emission_prob[tag][w] = (c+1)/total

###############################################
# UNKNOWN WORD PROBABILITY
###############################################

unknown_prob = {}

for tag in tags:
    total = sum(emission_counts[tag].values()) + len(words)
    unknown_prob[tag]=1/total

###############################################
# 9. LOG SPACE
###############################################

log_initial = np.log(initial_prob)

log_transition = np.log(transition_prob)

###############################################
# 10. VECTORIZED VITERBI
###############################################

def viterbi(sentence):

    T = len(sentence)

    dp = np.full((num_tags,T), -np.inf)

    backpointer = np.zeros((num_tags,T),dtype=int)

    ##################################################
    # FIRST WORD
    ##################################################

    emission = np.zeros(num_tags)

    for i,tag in enumerate(tags):

        emission[i]=np.log(
            emission_prob[tag].get(sentence[0],unknown_prob[tag])
        )

    dp[:,0]=log_initial+emission

    ##################################################
    # RECURSION
    ##################################################

    for t in range(1,T):

        emission=np.zeros(num_tags)

        for j,tag in enumerate(tags):

            emission[j]=np.log(
                emission_prob[tag].get(sentence[t],unknown_prob[tag])
            )

        scores = dp[:,t-1][:,None] + log_transition

        backpointer[:,t]=np.argmax(scores,axis=0)

        dp[:,t]=np.max(scores,axis=0)+emission

    ###############################################
    # BACKTRACK
    ###############################################

    best=np.argmax(dp[:,-1])

    path=[best]

    for t in range(T-1,0,-1):

        best=backpointer[best,t]

        path.append(best)

    path.reverse()

    return [idx2tag[i] for i in path]

###############################################
# 11. TESTING
###############################################

true_tags=[]

pred_tags=[]

for words_seq,tags_seq in test_sentences:

    prediction=viterbi(words_seq)

    true_tags.extend(tags_seq)

    pred_tags.extend(prediction)

###############################################
# 12. EVALUATION
###############################################

accuracy=accuracy_score(true_tags,pred_tags)

precision,recall,f1,_=precision_recall_fscore_support(
    true_tags,
    pred_tags,
    average='weighted'
)

print("\nAccuracy :",accuracy)

print("Precision:",precision)

print("Recall   :",recall)

print("F1 Score :",f1)

print("\nClassification Report\n")

print(classification_report(true_tags,pred_tags))

###############################################
# 13. TEST ON UNSEEN SENTENCES
###############################################

test_examples=[

"I love machine learning",

"She is reading a book",

"The weather is beautiful today",

"They will visit Chennai tomorrow",

"Artificial Intelligence changes the world"

]

for s in test_examples:

    words=s.split()

    prediction=viterbi(words)

    print("\nSentence:")

    print(words)

    print("Tags:")

    print(prediction)
