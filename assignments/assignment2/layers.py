import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    def softmax(predictions):
        if predictions.ndim == 1:
            pred = np.exp(predictions - np.max(predictions))
            probs = pred / np.sum(pred)
        else:
            pred = np.exp(predictions - np.max(predictions, axis=1)[:, None])
            probs = pred / np.sum(pred, axis=1)[:, None]
        return probs

    def cross_entropy_loss(probs, target_index):
        batch_size = np.array(target_index).size
        use_index = np.array(target_index).reshape(batch_size)
        if probs.ndim == 1:
            loss = -np.log(probs[use_index])
        else:
            loss = np.mean(-np.log(probs[np.arange(batch_size), use_index]))
        return loss

    probs = softmax(predictions)
    dprediction = probs
    loss = cross_entropy_loss(probs, target_index)
    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = np.array(target_index).size
        dprediction[np.arange(batch_size), target_index] -= 1
        dprediction /= batch_size

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.whereX = X > 0
        return np.where(X > 0, X, 0.)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.where(self.whereX, d_out, 0.)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_result = np.dot(d_out, self.W.value.T)
        batch_size = d_out.shape[0]
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1, batch_size)), d_out)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
