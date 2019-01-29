from nltk.corpus import treebank
import collections
tagged_sentences = treebank.tagged_sents(tagset='universal')
train = tagged_sentences[:3000]
test = tagged_sentences[3000:]

tagset = set([tag for sent in tagged_sentences for token, tag in sent])
tag2ids = {tag:id for id, tag in enumerate(tagset)}
word_counter = collections.Counter([token.lower() for sent in train for token, tag in sent])
vocab = [k for k , v in word_counter.items() if v > 3]
word2ids = {token:id+2 for id , token in enumerate(vocab)}
word2ids['<UNK'] = 0
word2ids['<PAD'] = 1
print(list(word2ids.items())[:10])
