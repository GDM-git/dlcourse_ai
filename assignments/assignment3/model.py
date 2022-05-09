import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = [
            ConvolutionalLayer(
                in_channels=input_shape[2],
                out_channels=input_shape[2],
                filter_size=conv1_channels,
                padding=2
            ),
            ReLULayer(),
            MaxPoolingLayer(
                pool_size=4,
                stride=2
            ),
            ConvolutionalLayer(
                in_channels=input_shape[2],
                out_channels=input_shape[2],
                filter_size=conv2_channels,
                padding=2
            ),
            ReLULayer(),
            MaxPoolingLayer(
                pool_size=4,
                stride=2
            ),
            Flattener(),
            FullyConnectedLayer(
                n_input=192,
                n_output=n_output_classes
            )
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        params = self.params()
        for param in params.values():
            param.grad.fill(0.0)

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        loss, dpred = softmax_with_cross_entropy(pred, y)

        for layer in reversed(self.layers):
            dpred = layer.backward(dpred)

        # for param in params.values():
            # loss_reg, grad_reg = l2_regularization(param.value, self.reg)
            # param.grad += grad_reg
            # loss += loss_reg

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        pred = np.argmax(pred, 1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters

        for i, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result.update({f'layer: {i}; param: {name}': param})

        return result
