
# coding: utf-8

# # Ngram lab
# 
# In this lab you will do 4 excercises building ngram language models:
# 1. A Maximum Liklihood Expectation (MLE) unigram model (10 marks)
# 2. A bigram model with add-one smoothing (10 marks)
# 3. A bigram model with general additive smoothing (10 marks)
# 4. (BONUS) A trigram model with Kneser-Ney snoothing (10 marks)
# 
# There are some examples using small corpus as seen in the lecture first, before you do the exercises using the following 3 files with line-separated text to train the bigger language models on:
# * training data -- "switchboard_lm_training.txt"
# * heldout data -- "switchboard_lm_heldout.txt"
# * test data -- "switchboard_lm_test.txt"

# In[ ]:


from __future__ import division  # for python 2 this is needed
from __future__ import print_function # for python 2 this is needed
from collections import Counter
from math import log


# In[ ]:


# Some useful methods
def glue_tokens(tokens, order):
    """A useful way of glueing tokens together for
    Kneser Ney smoothing and other smoothing methods
    
    :param: order is the order of the language model
        (1 = unigram, 2 = bigram, 3 =trigram etc.)
    """
    return '{0}@{1}'.format(order,' '.join(tokens))

def unglue_tokens(tokenstring, order):
    """Ungluing tokens glued by the glue_tokens method"""
    if order == 1:
        return [tokenstring.split("@")[1].replace(" ","")]
    return tokenstring.split("@")[1].split(" ")

def tokenize_sentence(sentence, order):
    """Returns a list of tokens with the correct numbers of initial
    and end tags (this is meant ot be used with a non-backoff model!!!)
    
    :sentence: a string of text
    :param: order is the order of the language model
        (1 = unigram, 2 = bigram, 3 =trigram etc.)
    """
    tokens = sentence.split()
    tokens = ['<s>'] * (order-1) + tokens + ['</s>']
    return tokens


# In[ ]:


#################################### Examples ############################
# Example set of sentences (corpus) from the lecture slides
sentences = [
            "I am Sam",
            "Sam I am",
            "I do not like green eggs and ham"
            ]


# In[ ]:


# Example 1. Build a unigram MLE language model from a simple corpus
unigrams = Counter()
for sent in sentences:
    words = tokenize_sentence(sent, 1)
    for w in words:
        unigrams[w] +=1
unigram_total = sum(unigrams.values())
# check that all add to one
check_if_adds_to_1 = 0
for k, v in unigrams.items():
    print(k, v/unigram_total)
    check_if_adds_to_1 += (v/unigram_total)
print("check if adds to 1:", check_if_adds_to_1)


# In[ ]:


# get the perplexity of those same sentences
# according to the model
# perplexity is always equal to two to the power of the entropy
# where entropy is the negative sum of all log probabilities from the model
N = 0 # total number of words
s = 0  # entropy
for sent in sentences:
    # get the unigram model based probability of each sentence
    words = tokenize_sentence(sent, 1)
    for w in words:
        N += 1
        prob = unigrams[w]/unigram_total
        logprob = log(prob, 2)  # get the log of the prob to base 2
        s += (-logprob)
perplexity = 2 ** (s/N)
print("cross entropy", s/N)
print("perplexity", perplexity)


# In[ ]:


# Example 2. Get probabilities for bigrams
bigrams = Counter()
bigram_context = Counter() # like unigrams, but the previous word only (so includes the start symbol)
delta = 1  # delta is order - 1
for s in sentences:
    words = tokenize_sentence(s, 2)
    for i in range(delta, len(words)):
        context = words[i-delta:i]
        target = words[i]
        ngram = context + [target]
        bigrams[glue_tokens(ngram, 2)] +=1
        bigram_context[glue_tokens(context, 1)] += 1
bigram_total = sum(bigrams.values())

# check if each bigram continuation sums to tomorrow
for context, v in bigram_context.items():
    context = unglue_tokens(context, 1)
    print("context", context)
    check_ngram_total_sums_to_1 = 0
    # for a given context the continuation probabilities 
    # over the whole vocab should sum to 1
    for u in unigrams.keys():
        ngram = context + [u]
        numerator = bigrams[glue_tokens(ngram, 2)] + 1
        denominator = v + (1 * len(bigram_context.items()))
        p = numerator / denominator
        # print(glue_tokens(ngram, 2), p)
        check_ngram_total_sums_to_1 += p
    print("check if sums to 1?", check_ngram_total_sums_to_1)


# In[ ]:


# Check the estimates for the lecture examples:
# p(I|<s>)
# p(Sam|<s>)
# p(am|I)
# p(</s>|Sam)
# p(Sam|am)
# p(do|I)

def bigram_MLE(ngram):
    """A simple function to compute the 
    MLE estimation based on the counters"""
    numerator = bigrams[glue_tokens(ngram, 2)]
    denominator = bigram_context[glue_tokens(ngram[:1], 1)]
    p = numerator / denominator
    return p

print(bigram_MLE(['<s>','I']))
print(bigram_MLE(['<s>', 'Sam']))
print(bigram_MLE(['I', 'am']))
print(bigram_MLE(['Sam', '</s>']))
print(bigram_MLE(['am', 'Sam']))
print(bigram_MLE(['I', 'do']))


# In[ ]:


# We we use the bigram and model to get the perplexity
# of each sentence
N = 0 # total number of words
s = 0 # entropy
for sent in sentences:
    words = tokenize_sentence(sent, 2)
    for i in range(delta, len(words)):
        N += 1
        context = words[i-delta:i]
        target = words[i]
        ngram = context + [target]
        numerator = bigrams.get(glue_tokens(ngram, 2)) 
        denominator = bigram_context.get(glue_tokens(context, 1))
        prob = numerator / denominator
        s += (-log(prob, 2))  # add the neg log prob
perplexity = 2 ** (s/N)
print("cross entropy", s/N)
print("perplexity", perplexity)


# # Exercises

# In[1]:


##############################################
# Exercise 1. Unigram MLE model from a bigger corpus
#
# Write code to read in the file 'switchboard_language_model_train.txt' which has preprocessed text on each line.
# Populate a unigram language model based on that data for an MLE estimation using a Counter (see Example 1 above).
# Keep updating the Counter for the model by reading in the data
# in 'switchboard_language_model_heldout.txt', however, this time include
# an unknown word token <unk/> for any words appearing in this data
# that were not in the first training data.
# Using this model, return the perplexity of the ENTIRE test corpus 'switchboard_lanaguage_model_test.txt', including
# replacing words unknown by the model with <unk/> to avoid not getting a perplexity score
##############################################


# In[3]:


##############################################
# Exercise 2. Bigram model with add-one smoothing
#
# Change your method for reading in and training a language model so it works for bigrams
# However, it should use add-one smoothing (see the lecture notes and Jurafsky & Martin Chapter 3)
# Remember this involves using the vocabulary size.
# Obtain the perplexity score on the test data as above for this bigram model
# Use the heldout corpus to get estimations for bigrams with unknown words as you did in Ex. 1.
##############################################


# In[ ]:


##############################################
# Exercise 3. Bigram model with general additive (Lidstone) smoothing
#
# Modify your code from exercise 2 such that it generalizes beyond
# adding 1 to all counts, but can add differing counts instead.
# Experiment with different values (e.g. 0.2, 0.4, 0.6, 0.8) and
# report the perplexity scores for all the different values you test.
# See if you can find the best amount to add.
##############################################


# In[ ]:


##############################################
# Exercise 4. Trigram model with Kneser-Ney smoothing
#
# Kneser-Ney smoothing is a state-of-the-art technique for smoothing n-gram models.
# The algorithm is quite complicated, and is implemented for you below for training
# on the training data (ngrams_interpolated_kneser_ney)
# The application at test time is done with the method kneser_ney_ngram_prob using the trained Counters.
# See if you can follow how it works, and refer to the below article on QM plus (pages 7-8 particularly):
# "A Bit of Progress in Language Modeling" - Joshua T. Goodman
#
# In this exercise, use the heldout data file to further train the model after you run the below
# but also replace words unseen in the first training data with the unknown word
# token <unk/>, as you should have done above.
# You do not need to modify the algorithms below, but just use them to train the Counters.
# Then obtain the perplexity score on the test data with your model and compare it to the other models.
# Experiment with different Discount weights (0.7 is used below, which works quite well),
# and even different values of n (e.g. 4-gram, 5-gram)
# to see if you can get the lowest possible perplexity score.
##############################################


# In[ ]:


# Kneser-Ney smoothing
order = 3
discount = 0.7

unigram_denominator = 0
ngram_numerator_map = Counter() 
ngram_denominator_map = Counter() 
ngram_non_zero_map = Counter()


def ngrams_interpolated_kneser_ney(tokens,
                                   order,
                                   ngram_numerator_map,
                                   ngram_denominator_map,
                                   ngram_non_zero_map,
                                   unigram_denominator):
    """This function counts the n-grams in tokens and also record the
    lower order non zero counts necessary for interpolated Kneser-Ney \
    smoothing,
    taken from Goodman 2001 and generalized to arbitrary orders"""
    for i in xrange(order-1,len(tokens)): # tokens should have a prefix of order - 1
        #print i
        for d in xrange(order,0,-1): #go through all the different 'n's
            if d == 1:
                unigram_denominator += 1
                ngram_numerator_map[glue_tokens(tokens[i],d)] += 1
            else:
                den_key = glue_tokens(tokens[i-(d-1) : i], d)
                num_key = glue_tokens(tokens[i-(d-1) : i+1], d)
    
                ngram_denominator_map[den_key] += 1
                # we store this value to check if it's 0
                tmp = ngram_numerator_map[num_key]
                ngram_numerator_map[num_key] += 1 # we increment it
                if tmp == 0: # if this is the first time we see this ngram
                    #number of types it's been used as a context for
                    ngram_non_zero_map[den_key] += 1
                else:
                    break 
                    # if the ngram has already been seen
                    # we don't go down to lower order models
    return ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator


# In[ ]:


# train the model
corpus = open("switchboard_lm_train.txt")
for line in corpus:
    tokens = tokenize_sentence(line, order)
    ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator =            ngrams_interpolated_kneser_ney(tokens,
                                           order,
                                           ngram_numerator_map,
                                           ngram_denominator_map,
                                           ngram_non_zero_map,
                                           unigram_denominator)
corpus.close() 


# In[ ]:


def kneser_ney_ngram_prob(ngram, discount, order):
    """KN smoothed ngram probability from Goodman 2001.
    This is run at test time.
    """
    tokens = []
    for token in ngram: 
        #put unknown token in for unknown words, only form of held 
        #out est used
        if (not ngram_numerator_map.get(glue_tokens(token,1)))         and not token =="<s>": #i.e. never seen at all
            tokens.append("<unk>")
        else:
            tokens.append(token)
    ngram = tokens #we've added our unk tokens

    #calculate the unigram prob of the last token 
    #as it appears as a numerator
    #if we've never seen it at all, it defacto will 
    #have no probability as a numerator
    uni_num = ngram_numerator_map.get(glue_tokens(ngram[-1],1))
    if uni_num == None:
        uni_num = 0
    probability = previous_prob = float(uni_num) / float(unigram_denominator)
    if probability == 0.0:
        print("0 prob!")
        print(glue_tokens(ngram[-1],1))
        print(ngram)
        print(ngram_numerator_map.get(glue_tokens(ngram[-1],1)))
        print(unigram_denominator)
        raise Exception

    # now we compute the higher order probs and interpolate
    for d in xrange(2,order+1):
        ngram_den = ngram_denominator_map.get(glue_tokens(ngram[-(d):-1],d))
        if ngram_den is None:
            ngram_den = 0
        #for bigrams this is the number of different continuation types
        # (number of trigram types with these two words)
        if ngram_den != 0: 
            #if this context (bigram, for trigrams) has never been seen, 
            #then we can only get unigram est, starts from two, goes up
            ngram_num = ngram_numerator_map.get(glue_tokens(ngram[-(d):],d)) 
            #this is adding one, use get?
            if ngram_num is None:
                ngram_num = 0
            if ngram_num != 0:
                current_prob = (ngram_num - discount) / float(ngram_den)
            else:
                current_prob = 0.0
            nonzero = ngram_non_zero_map.get(
                                    glue_tokens(ngram[-(d):-1],d))
            if nonzero is None:
                nonzero = 0
            current_prob += nonzero * discount / ngram_den * previous_prob
            previous_prob = current_prob
            probability = current_prob
        else:
            #current unseen contexts just give you the unigram 
            #back..not ideal.. we can learn <unk> from 
            #held out data though..
            probability = previous_prob
            break
    return probability

