import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input, hidden_layer_size))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(hidden_layer_size, n_output))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        params = self.params()
        for param in params.values():
            param.grad.fill(0.0)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        loss, dpred = softmax_with_cross_entropy(pred, y)

        for layer in reversed(self.layers):
            dpred = layer.backward(dpred)

        for param in params.values():
            loss_reg, grad_reg = l2_regularization(param.value, self.reg)
            param.grad += grad_reg
            loss += loss_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        pred = np.argmax(pred, 1)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        for i, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result.update({f'layer: {i}; param: {name}': param})

        return result
