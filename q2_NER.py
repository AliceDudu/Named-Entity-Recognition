import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init

import data_utils.utils as du
import data_utils.ner as ner

from utils import data_iterator
from model import LanguageModel

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  embed_size = 50	# 文字转化成向量的 L 的纬度
  batch_size = 64	# 每批处理x行数
  label_size = 5	# 类别数 y 的纬度
  hidden_size = 100	# 和assignment里面的一样
  max_epochs = 24 
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  l2 = 0.001
  window_size = 3	# 上下文窗口大小

class NERModel(LanguageModel):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.
  """

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    self.wv, word_to_num, num_to_word = ner.load_wv(
      'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
    tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    self.num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in self.num_to_tag.iteritems()}

    # Load the training set
    docs = du.load_dataset('data/ner/train')
    self.X_train, self.y_train = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    self.X_dev, self.y_dev = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    docs = du.load_dataset('data/ner/test.masked')
    self.X_test, self.y_test = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
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
        tf.int32, shape=[None, self.config.window_size], name='Input')
    self.labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, self.config.label_size], name='Target')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE
    feed_dict = {
        self.input_placeholder: input_batch,
    }
    if label_batch is not None:
      feed_dict[self.labels_placeholder] = label_batch
    if dropout is not None:
      feed_dict[self.dropout_placeholder] = dropout
    ### END YOUR CODE
    return feed_dict

  def add_embedding(self):
    """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    (Words are discrete entities. They need to be transformed into vectors for use
    in deep-learning. Although we won't do so in this problem, in practice it's
    useful to initialize the embedding with pre-trained word-vectors. For this
    problem, using the default initializer is sufficient.)

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: See following link to understand what -1 in a shape means.
      https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.wv), embed_size)

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      embedding = tf.get_variable('Embedding', [len(self.wv), self.config.embed_size])	# assignment中的 L	
      window = tf.nn.embedding_lookup(embedding, self.input_placeholder)				# 在 L 中直接把window大小的context的word vector搞定
      window = tf.reshape(
        window, [-1, self.config.window_size * self.config.embed_size])
      ### END YOUR CODE
      return window

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.							# 初始化assignment中有，Xavier initialization
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while				# 	cost function加上Regularization
          dropout should be added after both variable scopes.						# 	both variable scopes＝first hidden layer，softmax加上dropout
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you		# 用 tensorflow Graph 收集regularization and loss
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    ### YOUR CODE HERE
    with tf.variable_scope('Layer1', initializer=xavier_weight_init()) as scope:		# 用initializer=xavier去初始化第一层
      W = tf.get_variable(																# 第一层有 W，b1，h
          'W', [self.config.window_size * self.config.embed_size,
                self.config.hidden_size])
      b1 = tf.get_variable('b1', [self.config.hidden_size])
      h = tf.nn.tanh(tf.matmul(window, W) + b1)
      if self.config.l2:																# L2 regularization for W
          tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(W))	# 0.5 * self.config.l2 * tf.nn.l2_loss(W)

    with tf.variable_scope('Layer2', initializer=xavier_weight_init()) as scope:
      U = tf.get_variable('U', [self.config.hidden_size, self.config.label_size])
      b2 = tf.get_variable('b2', [self.config.label_size])
      y = tf.matmul(h, U) + b2
      if self.config.l2:
          tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(U))
    output = tf.nn.dropout(y, self.dropout_placeholder)									# 返回 output，两个variable_scope都带dropout
    ### END YOUR CODE
    return output 

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    cross_entropy = tf.reduce_mean(														# 1.关键步骤：loss是用cross entropy定义的
        tf.nn.softmax_cross_entropy_with_logits(y, self.labels_placeholder))				# y是模型预测值，计算cross entropy
    tf.add_to_collection('total_loss', cross_entropy)			# Stores value in the collection with the given name.
    															# collections are not sets, it is possible to add a value to a collection several times.
    loss = tf.add_n(tf.get_collection('total_loss'))			# Adds all input tensors element-wise. inputs: A list of Tensor with same shape and type
    ### END YOUR CODE
    return loss																			# 就是把所有的loss都加起来了，regular的和cross entropy的

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.			optimizer
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
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)    # 2.关键步骤：用 AdamOptimizer 使 loss 达到最小，所以更关键的是 loss
    ### END YOUR CODE
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data(debug=False)			# 1.关键步骤
    self.add_placeholders()
    window = self.add_embedding()	# window context的word vector
    y = self.add_model(window)		# 1.关键步骤：经过2层后的output，只是线性处理完 hU＋b2

    self.loss = self.add_loss_op(y)			# 1.关键步骤：对y预测和实际，计算cross entropy
    self.predictions = tf.nn.softmax(y)			# 4.关键步骤：对y用softmax
    one_hot_prediction = tf.argmax(self.predictions, 1)			# 找到预测的类别的位置
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)		# 看预测和实际的两个位置是否相等
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))			# 3，关键步骤：将correct转换type，然后求和计算多少多的
    self.train_op = self.add_training_op(self.loss)				# 2.关键步骤：把loss用optimizer最小化

  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels			# 定义数据 x，y，dropout
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0							# 计算分类准确的比例：correct／total processed
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size	# X的总size／每批的size 得到批次数
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(											# 前面都是定义，这里才是执行	
          [self.loss, self.correct_predictions, self.train_op],		# 1.喂进去x y dropout，执行 loss，correct_predictions，train_op
          feed_dict=feed)															# train后，参数W等就已经被更新了
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)		# 返回 mean(loss) 和 准确度

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)								# 把X和dropout喂进去，执行loss和prediction
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)		# 1.没有y的时候，喂进去x，执行 loss 和 predictions
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results													# 返回 loss 和 extend(argmax)

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write(str(prediction) + "\n")

def test_NER():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = NERModel(config)   # 最主要的类

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')  # 最好的值时，它的 loss 它的 迭代次数 epoch？
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train)   # 1.把 train 数据放进迭代里跑，得到 loss 和 accuracy
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)   # 2.用这个model去预测 dev 数据，得到loss 和 prediction
        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        if val_loss < best_val_loss:			# 用 val 数据的loss去找最小的loss
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")
        
          saver.save(session, './weights/ner.weights')   # 把最小的 loss 对应的 weights 保存起来
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
        confusion = calculate_confusion(config, predictions, model.y_dev)  # 3.把 dev 的lable数据放进去，计算prediction的confusion
        print_confusion(confusion, model.num_to_tag)
        print 'Total time: {}'.format(time.time() - start)
      
      saver.restore(session, './weights/ner.weights')	# 再次加载保存过的 weights，用 test 数据做预测，得到预测结果
      print 'Test'
      print '=-=-='
      print 'Writing predictions to q2_test.predicted'
      _, predictions = model.predict(session, model.X_test, model.y_test)
      save_predictions(predictions, "q2_test.predicted")	# 把预测结果保存起来

if __name__ == "__main__":
  test_NER()
