import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment

    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment

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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment

        self.whereX = X > 0
        return np.where(X > 0, X, 0.)

    def backward(self, d_out):
        # TODO copy from the previous assignment

        d_result = np.where(self.whereX, d_out, 0.)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment

        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment

        d_result = np.dot(d_out, self.W.value.T)
        batch_size = d_out.shape[0]
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1, batch_size)), d_out)

        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))

        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        X_with_pad = np.zeros((batch_size , height + 2 * self.padding , width + 2 * self.padding , channels))
        X_with_pad[:, self.padding : height + self.padding, self.padding : width + self.padding, :] = X
        self.X = X_with_pad

        out_height  = X_with_pad.shape[1] - self.filter_size + 1
        out_width = X_with_pad.shape[2] - self.filter_size + 1
        out = np.zeros((batch_size , out_height  , out_width , self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location

                out[:, y : y + self.filter_size, x : x + self.filter_size, :] = (
                    np.dot(
                        X_with_pad[:, y : y + self.filter_size, x : x + self.filter_size, :].reshape(batch_size, -1),
                        self.W.value.reshape(-1, self.out_channels)
                    ) + self.B.value
                ).reshape(batch_size, 1, 1, self.out_channels)
        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_in = np.zeros(self.X.shape)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                in_local = self.X[:, y : y + self.filter_size, x : x + self.filter_size, :]
                d_out_local = d_out[:, y : y + 1, x : x + 1, :].reshape(batch_size, -1)

                d_in[:, y : y + self.filter_size, x : x + self.filter_size, :] += np.dot(
                    d_out_local,
                    self.W.value.reshape(-1, self.out_channels).T
                ).reshape(in_local.shape)

                self.W.grad += np.dot(
                    in_local.reshape(batch_size,-1).T,
                    d_out_local
                ).reshape(self.W.value.shape)

                self.B.grad += np.dot(
                    np.ones((1, batch_size)),
                    d_out_local
                ).reshape(self.B.value.shape)

        return d_in[:, self.padding : height - self.padding, self.padding : width - self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        self.X = X
        self.masks.clear()

        res_height  = int((height - self.pool_size) / self.stride + 1)
        res_width = int((width - self.pool_size) / self.stride + 1)

        out = np.zeros((batch_size, res_height, res_width, channels))

        for x in range(res_width):
            for y in range(res_height):
                x_in_use, y_in_use = x * self.stride, y * self.stride
                in_local = X[:, y_in_use : y_in_use + self.pool_size, x_in_use : x_in_use + self.pool_size, :]
                out[:, y, x, :] = np.max(in_local, (1, 2))

                mask = np.zeros_like(in_local)
                xy_idx = np.argmax(in_local.reshape(batch_size, -1, channels), 1)
                batch_idx, channel_idx = np.indices((batch_size, channels))
                mask.reshape(batch_size, -1, channels)[batch_idx, xy_idx, channel_idx] = 1
                self.masks[(x, y)] = mask

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        
        _, res_height, res_width, _ = d_out.shape
        d_in = np.zeros(self.X.shape)

        for x in range(res_width):
            for y in range(res_height):
                x_in_use, y_in_use = x * self.stride, y * self.stride
                d_in[
                    :,
                    y_in_use : y_in_use + self.pool_size,
                    x_in_use : x_in_use + self.pool_size,
                    :
                ] += self.masks[(x, y)] * d_out[:, y : y + 1, x : x + 1, :]

        return d_in

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]

        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass

        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
