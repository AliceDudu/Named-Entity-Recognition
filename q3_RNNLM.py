import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64			# 分批处理
  embed_size = 50			# 转化成词向量
  hidden_size = 100
  num_steps = 10
  max_epochs = 16			# 迭代次数
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],			# 将句子get成word，再encode成one－hot向量
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(
        tf.int32, shape=[None, self.config.num_steps], name='Input')			
    self.labels_placeholder = tf.placeholder(
        tf.int32, shape=[None, self.config.num_steps], name='Target')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):														
      ### YOUR CODE HERE
      embedding = tf.get_variable(
          'Embedding',
          [len(self.vocab), self.config.embed_size], trainable=True)				# L: (len(self.vocab), embed_size)
      inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)			# Looks up ids in a list of embedding tensors.
      inputs = [
          tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]	# remove specific dimensions of size 1 at postion=[1]
      ### END YOUR CODE
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    with tf.variable_scope('Projection'):
      U = tf.get_variable(
          'Matrix', [self.config.hidden_size, len(self.vocab)])
      proj_b = tf.get_variable('Bias', [len(self.vocab)])						
      outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]					# outputs＝rnn_outputs＊U＋b2
    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss. 

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
    cross_entropy = sequence_loss(												# cross entropy
        [output], [tf.reshape(self.labels_placeholder, [-1])], all_ones, len(self.vocab))
    tf.add_to_collection('total_loss', cross_entropy)
    loss = tf.add_n(tf.get_collection('total_loss'))							# 最终的loss
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    optimizer = tf.train.AdamOptimizer(self.config.lr)							
    train_op = optimizer.minimize(self.calculate_loss)							# 用Adam最小化loss
    ### END YOUR CODE
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)								# rnn网络
    self.outputs = self.add_projection(self.rnn_outputs)						# 对rnn输出结果进行projection
  
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]			# 对projection进行softmax
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])		# 对outputs进行reshape得到output
    self.calculate_loss = self.add_loss_op(output)								# 对output计算loss
    self.train_step = self.add_training_op(self.calculate_loss)					# 训练，使loss达到最小


  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable		# initial state for the RNN

          self.initial_state
  
          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable					# last RNN output

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.					dropout
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:
      
          H: (hidden_size, hidden_size) 
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    with tf.variable_scope('InputDropout'):
      inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]			# dropout of inputs

    with tf.variable_scope('RNN') as scope:
      self.initial_state = tf.zeros(												# initial state of RNN
          [self.config.batch_size, self.config.hidden_size])
      state = self.initial_state
      rnn_outputs = []
      for tstep, current_input in enumerate(inputs):								# tstep 多少个时刻，多少个单词
        if tstep > 0:
          scope.reuse_variables()
        RNN_H = tf.get_variable(
            'HMatrix', [self.config.hidden_size, self.config.hidden_size])			
        RNN_I = tf.get_variable(
            'IMatrix', [self.config.embed_size, self.config.hidden_size])
        RNN_b = tf.get_variable(
            'B', [self.config.hidden_size])
        state = tf.nn.sigmoid(
            tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)		# 看这里state应该是当前时刻的隐藏层
        rnn_outputs.append(state)													# 不过它在下一个循环中就被用了，所以也是用来存上一时刻隐藏层的
      self.final_state = rnn_outputs[-1]

    with tf.variable_scope('RNNDropout'):
      rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]		# dropout of outputs
    ### END YOUR CODE
    return rnn_outputs


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)			# 用RNN的final state，计算loss，并用train训练到最小loss
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))							# 存step和loss，存到哪里啦？weights呢？
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))													# 返回loss的指数平均

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.									# 输入text，输出List of word idxs，来generate_text
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]				# 把输入text分解成word，再转化成one hot向量
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    feed = {model.input_placeholder: [tokens[-1:]],
            model.initial_state: state,
            model.dropout_placeholder: 1}
    state, y_pred = session.run(
        [model.final_state, model.predictions[-1]], feed_dict=feed)					# 用model去预测，得到state, y_pred
    ### END YOUR CODE
    next_word_idx = sample(y_pred[0], temperature=temp)								# 下一个单词在词库里的位置idx
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]					# 将tokens里的one hot向量解码成word
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)	# 生成文本，stoplength=100停止

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)															# 要训练的model
    # This instructs gen_model to reuse the same variables as the model above
    scope.reuse_variables()
    gen_model = RNNLM_Model(gen_config)													# 要reuse的model

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    for epoch in xrange(config.max_epochs):										# 迭代max epoch次
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)					# 代入encoded train和valid数据，训练model，得到perplexity
      print 'Training perplexity: {}'.format(train_pp)							# training data和validation data的 perplexity
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')								# 选择最小的 valid perplexity 并保存相应的weights
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
      
    saver.restore(session, 'ptb_rnnlm.weights')
    test_pp = model.run_epoch(session, model.encoded_test)						# model.run_epoch，训练这个model
    print '=-=' * 5																
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    starting_text = 'in palo alto'
    while starting_text:
      print ' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0))	# 用模型作用在输入的初始文本，生成后面的单词
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
