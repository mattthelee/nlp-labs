import tensorflow as tf
import collections
import time
from nltk.translate.bleu_score import corpus_bleu

tf.reset_default_graph()


class NmtModel(object):
  def __init__(self,source_dict,target_dict,use_attention):
    self.num_layers = 2
    self.hidden_size = 200
    self.embedding_size = 100
    self.hidden_dropout_rate=0.2
    self.embedding_dropout_rate = 0.2
    self.max_target_step = 30
    self.vocab_target_size = len(target_dict.vocab)
    self.vocab_source_size = len(source_dict.vocab)
    self.target_dict = target_dict
    self.source_dict = source_dict
    self.SOS = target_dict.word2ids['<start>']
    self.EOS = target_dict.word2ids['<end>']
    self.use_attention = use_attention

    print("source vocab: %d, target vocab:%d" % (self.vocab_source_size,self.vocab_target_size))


  def build(self):
    self.source_words = tf.placeholder(tf.int32,[None,None],"source_words")
    self.target_words = tf.placeholder(tf.int32,[None,None],"target_words")
    self.source_sent_lens = tf.placeholder(tf.int32,[None],"source_sent_lens")
    self.target_sent_lens = tf.placeholder(tf.int32,[None],"target_sent_lens")
    self.is_training = tf.placeholder(tf.bool,[],"is_training")

    self.predictions,self.loss = self.get_predictions_and_loss(self.source_words,self.target_words,self.source_sent_lens,self.target_sent_lens,self.is_training)

    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params))
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())


  def get_predictions_and_loss(self, source_words,target_words, source_sent_lens,target_sent_lens,is_training):
    self.embeddings_target = tf.get_variable("embeddings_target", [self.vocab_target_size, self.embedding_size], dtype=tf.float32)
    self.embeddings_source = tf.get_variable("embeddings_source", [self.vocab_source_size, self.embedding_size], dtype=tf.float32)

    batch_size = shape(target_words, 0)
    max_target_sent_len = shape(target_words, 1)

    embedding_keep_prob = 1 - (tf.to_float(is_training) * self.embedding_dropout_rate)
    hidden_keep_prob = 1 - (tf.to_float(is_training) * self.hidden_dropout_rate)

    source_embs = tf.nn.dropout(tf.nn.embedding_lookup(self.embeddings_source,source_words),embedding_keep_prob)
    target_embs = tf.nn.dropout(tf.nn.embedding_lookup(self.embeddings_target,target_words),embedding_keep_prob)


    encoder_outputs, encode_final_states = self.encoder(source_embs,source_sent_lens,hidden_keep_prob)

    time_major_target_embs = tf.transpose(target_embs,[1,0,2])


    def _decoder_scan(pre,inputs):
      pre_logits, pre_pred, pre_states = pre
      step_embeddings = inputs

      pred_embeddings = tf.nn.embedding_lookup(self.embeddings_target,pre_pred)

      step_embeddings = tf.cond(is_training,lambda :step_embeddings,lambda :pred_embeddings)
      curr_logits, curr_states = self.step_decoder(step_embeddings,encoder_outputs,pre_states,hidden_keep_prob)
      curr_pred = tf.argmax(curr_logits,1,output_type=tf.int32)

      return curr_logits, curr_pred, curr_states

    init_logits = tf.zeros([batch_size,self.vocab_target_size])
    init_pred = tf.ones([batch_size],tf.int32) * self.SOS

    time_major_logits, time_major_preds, _ = tf.scan(_decoder_scan,time_major_target_embs,initializer=(init_logits, init_pred,encode_final_states))
    time_major_logits, time_major_preds = tf.stack(time_major_logits),tf.stack(time_major_preds)

    logits = tf.transpose(time_major_logits,[1,0,2])
    predictions = tf.transpose(time_major_preds,[1,0])

    logits_mask = tf.sequence_mask(target_sent_lens-1,max_target_sent_len)
    flatten_logits_mask = tf.reshape(logits_mask,[batch_size*max_target_sent_len])
    flatten_logits = tf.boolean_mask(tf.reshape(logits,[batch_size*max_target_sent_len,self.vocab_target_size]),flatten_logits_mask)

    gold_labels_mask = tf.concat([tf.zeros([batch_size,1],dtype=tf.bool),tf.sequence_mask(target_sent_lens-1,max_target_sent_len-1)],1)
    flatten_gold_labels_mask = tf.reshape(gold_labels_mask,[batch_size*max_target_sent_len])
    flatten_gold_labels = tf.boolean_mask(tf.reshape(target_words,[batch_size*max_target_sent_len]),flatten_gold_labels_mask)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flatten_gold_labels,logits=flatten_logits))

    return predictions, loss



    def encoder(self,embeddings, sent_lens, hidden_keep_prob=1.0):
        with tf.variable_scope("encoder"):
            """
            Task 1 encoder

            Start
            """

            word_embeddings = tf.nn.dropout(embeddings,embedding_keep_prob)
            word_lstm_first = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                          state_keep_prob=hidden_keep_prob,
                                                          variational_recurrent=True,
                                                          dtype=tf.float32)
            word_lstm_second = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                          state_keep_prob=hidden_keep_prob,
                                                          variational_recurrent=True,
                                                          dtype=tf.float32)
            lstm_cells = tf.nn.rnn_cell.MultiRNNCell([word_lstm_first, word_lstm_second])
            output,state = tf.nn.dynamic_rnn(lstm_cells, word_embeddings,sequence_length=sent_lens, dtype=tf.float32)

            """
            End Task 1
            """
            return output, state

    return encoder_outputs, encoder_final_states


  def step_decoder(self,step_embeddings,encoder_outputs, pre_states, hidden_keep_prob=1.0):
    with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
        """
        Task 2 decoder without attention

        Start
        """
        word_embeddings = tf.nn.dropout(step_embeddings,embedding_keep_prob)
        word_lstm_first = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                      state_keep_prob=hidden_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32)
        word_lstm_second = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                      state_keep_prob=hidden_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([word_lstm_first, word_lstm_second])
        output,state = tf.nn.dynamic_rnn(lstm_cells, word_embeddings,sequence_length=1, dtype=tf.float32)

        """
        End Task 2
        """

        """
        Task 3 attention

        Start
        """





        """
        Ends Task 3
        """

    return logits, curr_states



  def time_used(self, start_time):
    curr_time = time.time()
    used_time = curr_time-start_time
    m = used_time // 60
    s = used_time - 60 * m
    return "%d m %d s" % (m, s)

  def train(self,train_data,dev_data,test_data, epochs):
    start_time = time.time()
    for epoch in range(epochs):
      print("Starting training epoch {}/{}".format(epoch + 1, epochs))
      epoch_time = time.time()
      losses = []
      source_train,target_train = train_data
      for i, (source,target) in enumerate(zip(source_train,target_train)):
        source_words,source_sent_lens = source
        target_words,target_sent_lens = target
        fd = {self.source_words:source_words,self.target_words:target_words,
              self.source_sent_lens:source_sent_lens,self.target_sent_lens:target_sent_lens,
              self.is_training:True}

        _, loss= self.sess.run([self.train_op, self.loss], feed_dict=fd)

        losses.append(loss)
        if (i+1) % 100 == 0:
          print("[{}]: loss:{:.2f}".format(i+1, sum(losses[i + 1 - 100:]) / 100.0))
      print("Average epoch loss:{}".format(sum(losses) / len(losses)))
      print("Time used for epoch {}: {}".format(epoch + 1, self.time_used(epoch_time)))
      dev_time = time.time()
      print("Evaluating on dev set after epoch {}/{}:".format(epoch + 1, epochs))
      self.eval(dev_data)
      print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

    print("Training finished!")
    print("Time used for training: {}".format(self.time_used(start_time)))

    print("Evaluating on test set:")
    test_time = time.time()
    self.eval(test_data)
    print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))



  def get_target_sentences(self, sents,vocab,reference=False,isnumpy=False):
    str_sents = []
    for sent in sents:
      str_sent = []
      for t in sent:
        if isnumpy:
          t = t.item()
        if t == self.SOS:
          continue
        if t == self.EOS:
          break

        str_sent.append(vocab[t])
      if reference:
        str_sents.append([str_sent])
      else:
        str_sents.append(str_sent)
    return str_sents


  def eval(self, dataset):
    source_batches, target_batches = dataset
    references = []
    candidates = []
    vocab = self.target_dict.vocab
    PAD = self.target_dict.PAD

    for i, (source, target) in enumerate(zip(source_batches, target_batches)):
      source_words, source_sent_lens = source
      target_words, target_sent_lens = target
      infer_target_words = [[PAD for i in range(self.max_target_step)] for b in target_words]

      fd = {self.source_words: source_words, self.target_words: infer_target_words,
            self.source_sent_lens: source_sent_lens,
            self.is_training: False}
      predictions = self.sess.run(self.predictions,feed_dict=fd)

      references.extend(self.get_target_sentences(target_words,vocab,reference=True))
      candidates.extend(self.get_target_sentences(predictions,vocab,isnumpy=True))

    score = corpus_bleu(references,candidates)
    print("Model BLEU score: %.2f" % (score*100.0))



def shape(x, n):
  return x.get_shape()[n].value or tf.shape(x)[n]

class LanguageDict():
  def __init__(self, sents):
    word_counter = collections.Counter(tok.lower() for sent in sents for tok in sent)

    self.vocab = [t for t,c in word_counter.items() if c > 10]
    self.vocab.append('<pad>')
    self.vocab.append('<unk>')
    self.word2ids = {w:id for id, w in enumerate(self.vocab)}
    self.UNK = self.word2ids['<unk>']
    self.PAD = self.word2ids['<pad>']


def load_dataset(path, max_num_examples=30000,batch_size=100,add_start_end = False):
  lines = [line for line in open(path,'r')]
  if max_num_examples > 0:
    max_num_examples = min(len(lines), max_num_examples)
    lines = lines[:max_num_examples]

  sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in lines]
  if add_start_end:
    for sent in sents:
      sent.append('<end>')
      sent.insert(0,'<start>')

  lang_dict = LanguageDict(sents)

  sents = [[lang_dict.word2ids.get(tok,lang_dict.UNK) for tok in sent] for sent in sents]

  batches = []
  for i in range(len(sents) // batch_size):
    batch = sents[i * batch_size:(i + 1) * batch_size]
    batch_len = [len(sent) for sent in batch]
    max_batch_len = max(batch_len)
    for sent in batch:
      if len(sent) < max_batch_len:
        sent.extend([lang_dict.PAD for _ in range(max_batch_len - len(sent))])
    batches.append((batch, batch_len))


  unit = len(batches)//10
  train_batches = batches[:8*unit]
  dev_batches = batches[8*unit:9*unit]
  test_batches = batches[9*unit:]

  return train_batches,dev_batches,test_batches,lang_dict



if __name__ == '__main__':
  batch_size = 100
  max_example = 30000
  use_attention = True
  source_train, source_dev, source_test, source_dict = load_dataset("data.30.vi",max_num_examples=max_example,batch_size=batch_size)
  target_train, target_dev, target_test, target_dict = load_dataset("data.30.en", max_num_examples=max_example,batch_size=batch_size, add_start_end=True)
  print("read %d/%d/%d train/dev/test batches" % (len(source_train),len(source_dev), len(source_test)))

  train_data = (source_train,target_train)
  dev_data = (source_dev,target_dev)
  test_data = (source_test,target_test)

  model = NmtModel(source_dict,target_dict,use_attention)
  model.build()
  model.train(train_data,dev_data,test_data,10)
