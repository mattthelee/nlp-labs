import metrics,time
import json
import numpy as np
import tensorflow as tf
import collections

class CorefModel(object):
  def __init__(self,embedding_path, embedding_size):
    self.embedding_path = embedding_path
    self.embedding_size = embedding_size
    self.embedding_dropout_rate = 0.5
    self.max_ant = 250
    self.hidden_size = 150
    self.ffnn_layer = 2
    self.hidden_dropout_rate = 0.2

  def build(self):
    self.embedding_dict = self.load_embeddings(self.embedding_path, self.embedding_size)

    self.word_embeddings = tf.placeholder(tf.float32, shape=[None, None,self.embedding_size])
    self.sent_lengths = tf.placeholder(tf.int32, shape=[None])
    self.mention_starts = tf.placeholder(tf.int32, shape=[None])
    self.mention_ends = tf.placeholder(tf.int32, shape=[None])
    self.mention_cluster_ids = tf.placeholder(tf.int32, shape=[None])
    self.is_training = tf.placeholder(tf.bool, shape=[])

    self.predictions, self.loss = self.get_predictions_and_loss(
      self.word_embeddings,self.sent_lengths,self.mention_starts,self.mention_ends,
      self.mention_cluster_ids,self.is_training)

    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients,5.0)
    optimizer = tf.train.AdamOptimizer()
    self.train_op = optimizer.apply_gradients(zip(gradients,trainable_params))
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def get_predictions_and_loss(self,word_embeddings,sent_lengths,mention_starts,mention_ends,mention_cluster_ids,is_training):
    embedding_keep_prob = 1 - (tf.to_float(is_training)*self.embedding_dropout_rate)
    hidden_keep_prob = 1 - (tf.to_float(is_training)*self.hidden_dropout_rate)


    """
    Task 1: Create a bidirectional LSTM
    
    Begin
    """

    


    """
    Task 1 End
    """

    word_output = tf.concat([output_for, output_rev], axis=-1)

    num_sents = self.shape(word_embeddings, 0)
    max_sent_length = self.shape(word_embeddings, 1)

    word_seq_mask = tf.sequence_mask(sent_lengths, max_sent_length)
    flatten_word_seq_mask = tf.reshape(word_seq_mask, [num_sents * max_sent_length])
    flatten_word_output = tf.reshape(word_output, [num_sents * max_sent_length, 2 * self.hidden_size])
    flatten_word_output = tf.nn.dropout(flatten_word_output, hidden_keep_prob)
    flatten_word_output = tf.boolean_mask(flatten_word_output, flatten_word_seq_mask, axis=0)

    mention_starts_emb = tf.gather(flatten_word_output,mention_starts)
    mention_ends_emb = tf.gather(flatten_word_output,mention_ends)

    mention_emb = tf.concat([mention_starts_emb,mention_ends_emb],axis=1)

    num_mention = self.shape(mention_emb, 0)
    max_ant = tf.minimum(num_mention,self.max_ant)

    antecedents = tf.expand_dims(tf.range(num_mention),1) \
                  - tf.tile(tf.expand_dims(tf.range(max_ant)+1, 0), [num_mention, 1])
    antecedents_mask = antecedents >= 0
    antecedents = tf.maximum(antecedents, 0)
    antecedents_emb = tf.gather(mention_emb, antecedents)

    tiled_mention_emb = tf.tile(tf.expand_dims(mention_emb, 1), [1,max_ant,1])

    mention_pair_emb = tf.concat([tiled_mention_emb, antecedents_emb], 2)

    ffnn_input = tf.reshape(mention_pair_emb,[num_mention*max_ant, 8 * self.hidden_size])


    """
    Task 2: Create a multilayer feed-forward neural network.
    
    Begin
    """


    """
    Task 2 End
    """

    mention_pair_scores = tf.reshape(mention_pair_scores,[num_mention,max_ant])
    mention_pair_scores += tf.log(tf.to_float(antecedents_mask))

    dummy_scores = tf.zeros([num_mention,1])

    mention_pair_scores = tf.concat([dummy_scores,mention_pair_scores], 1)

    antecedents_cluster_ids = tf.gather(mention_cluster_ids,antecedents) + tf.to_int32(tf.log(tf.to_float(antecedents_mask)))
    mention_pair_labels = tf.logical_and(
      tf.equal(antecedents_cluster_ids, tf.expand_dims(mention_cluster_ids, 1)),
      tf.greater(antecedents_cluster_ids, 0))
    dummy_labels = tf.logical_not(tf.reduce_any(mention_pair_labels,1,keepdims=True))
    mention_pair_labels = tf.concat([dummy_labels,mention_pair_labels],1)

    gold_scores = mention_pair_scores + tf.log(tf.to_float(mention_pair_labels))
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores,1)
    log_norm = tf.reduce_logsumexp(mention_pair_scores,1)
    loss = log_norm - marginalized_gold_scores
    loss = tf.reduce_sum(loss)

    return [antecedents, mention_pair_scores], loss

  def shape(self, x, n):
    return x.get_shape()[n].value or tf.shape(x)[n]

  def load_embeddings(self, path, size):
    print("Loading word embeddings from {}...".format(path))
    embeddings = collections.defaultdict(lambda: np.zeros(size))
    for line in open(path):
      splitter = line.find(' ')
      emb = np.fromstring(line[splitter + 1:], np.float32, sep=' ')
      assert len(emb) == size
      embeddings[line[:splitter]] = emb
    print("Finished loading word embeddings")
    return embeddings

  def get_feed_dict_list(self,path, is_training):
    feed_dict_list = []
    for line in open(path):
      doc = json.loads(line)

      clusters = doc['clusters']
      gold_mentions = sorted([tuple(m) for cl in clusters for m in cl])
      gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
      cluster_ids = np.zeros(len(gold_mentions))
      for cid, cluster in enumerate(clusters):
        for mention in cluster:
          cluster_ids[gold_mention_map[tuple(mention)]] = cid + 1

      starts, ends = [], []
      if len(gold_mentions) > 0:
        starts, ends = zip(*gold_mentions)

      starts, ends = np.array(starts), np.array(ends)

      sentences = doc['sentences']
      sent_lengths = [len(sent) for sent in sentences]
      max_sent_length = max(sent_lengths)
      word_emb = np.zeros([len(sentences),max_sent_length,self.embedding_size])
      for i, sent in enumerate(sentences):
        for j, word in enumerate(sent):
          word_emb[i,j] = self.embedding_dict[word]




      fd = {}
      fd[self.word_embeddings] = word_emb
      fd[self.sent_lengths] = np.array(sent_lengths)
      fd[self.mention_starts] = starts
      fd[self.mention_ends] = ends
      fd[self.mention_cluster_ids] = cluster_ids
      fd[self.is_training] = is_training
      feed_dict_list.append(tuple((fd,clusters)))

    return feed_dict_list


  def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []

    """
    Task 3: Form the predicted clusters.
    
    Begin
    """


    """
    Task 3 End
    """

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, mention_starts, mention_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends,
                                                                           predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

  def train(self, train_path,dev_path,test_path, epochs):
    train_fd_list = self.get_feed_dict_list(train_path, True)
    start_time = time.time()
    for epoch in xrange(epochs):
      print("Starting training epoch {}/{}".format(epoch+1,epochs))
      epoch_time = time.time()
      losses = []
      for i, (fd, _) in enumerate(train_fd_list):
        _,loss = self.sess.run([self.train_op,self.loss], feed_dict=fd)
        losses.append(loss)
        if i>0 and i%200 == 0:
          print("[{}]: loss:{:.2f}".format(i,sum(losses[i-200:])/200.0))
      print("Average epoch loss:{}".format(sum(losses)/len(losses)))
      print("Time used for epoch {}: {}".format(epoch+1, self.time_used(epoch_time)))
      dev_time = time.time()
      print("Evaluating on dev set after epoch {}/{}:".format(epoch+1,epochs))
      self.eval(dev_path)
      print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

    print("Training finished!")
    print("Time used for training: {}".format(self.time_used(start_time)))

    print("Evaluating on test set:")
    test_time = time.time()
    self.eval(test_path)
    print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))


  def eval(self, path):
    eval_fd_list = self.get_feed_dict_list(path, False)
    coref_evaluator = metrics.CorefEvaluator()

    for fd, clusters in eval_fd_list:
      mention_starts,mention_ends = fd[self.mention_starts],fd[self.mention_ends]
      antecedents, mention_pair_scores = self.sess.run(self.predictions, fd)

      predicted_antecedents = []
      for i, index in enumerate(np.argmax(mention_pair_scores, axis=1) - 1):
        if index < 0:
          predicted_antecedents.append(-1)
        else:
          predicted_antecedents.append(antecedents[i, index])

      self.evaluate_coref(mention_starts,mention_ends,predicted_antecedents,clusters,coref_evaluator)

    p, r, f = coref_evaluator.get_prf()
    print("Average F1 (py): {:.2f}%".format(f * 100))
    print("Average precision (py): {:.2f}%".format(p * 100))
    print("Average recall (py): {:.2f}%".format(r * 100))

  def time_used(self, start_time):
    curr_time = time.time()
    used_time = curr_time-start_time
    m = used_time // 60
    s = used_time - 60 * m
    return "%d m %d s" % (m, s)

if __name__ == '__main__':
  embedding_path = 'glove.840B.300d.txt.filtered'
  train_path = 'train.english.30sent.jsonlines'
  dev_path = 'dev.english.jsonlines'
  test_path = 'test.english.jsonlines'
  embedding_size = 300
  model = CorefModel(embedding_path,embedding_size)
  model.build()
  model.train(train_path,dev_path,test_path,5)