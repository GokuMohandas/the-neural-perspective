import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    
    scores = X[i, :].dot(W)
    scores -= np.max(scores)
    normalized_scores = np.exp(scores) / np.sum(np.exp(scores))  
    
    for j in xrange(num_classes):

        if j == y[i]:
            loss += -np.log( normalized_scores[j] )
            dW[:, j] += (normalized_scores[j] - 1.0) * X[i, :]
        else:
            dW[:, j] += (normalized_scores[j] - 0.0) * X[i, :]
   
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.exp(X.dot(W))
  normalized_scores = (scores.T / np.sum(scores, axis=1)).T # division is only by axis=0 so we have to use transpose trick
 
  ground_truth = np.zeros_like(normalized_scores)
  ground_truth[range(num_train), y] = 1.0 # correct class

  loss = np.sum(np.sum(-np.log(normalized_scores[range(num_train), y])))
  dW = X.T.dot(normalized_scores - ground_truth)
  
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW