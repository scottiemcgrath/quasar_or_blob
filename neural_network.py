from more_itertools import pairwise
import numpy as np
from nose import tools as nt                         # for testing
from sklearn.datasets import load_digits             # for testing
from sklearn import preprocessing                    # for testing
from sklearn.model_selection import train_test_split # for testing
import matplotlib.pyplot as plt

class Sigmoid:
    # the sigmoid activation function
    # use with binary cross entropy loss or softmax and categorical cross entropy
    # susceptible to dying gradients
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return 1.0 / (1.0 + np.exp(-z_vectors))
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return Sigmoid.magnitude(z_vectors) * (1 - Sigmoid.magnitude(z_vectors))

class Tanh:
    # the tanh activation function
    # possibly a better version of sigmoid, though we're not really sure
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return (2.0 / (1.0 + np.exp(-2 * z_vectors))) - 1
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return 1 - Tanh.magnitude(z_vectors)**2

class Arctan:
    # the arctan ativation function
    # similar to tanh
    # less susceptible to dying gradients, but unbounded
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return np.arctan(z_vectors)
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return 1 / (z_vectors**2 + 1)

class Relu:
    # the relu activation function
    # very fast to compute
    # no dying gradients problem
    # susceptible to dying relu problem
    # use with mean squared error cost function
    # not everywhere differentiable
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return np.where(z_vectors<=0, 0, z_vectors)
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return np.where(z_vectors<=0, 0, 1)

class LeakyRelu:
    # the leaky relu activation function
    # similar to relu but less susceptible to the dying relu problem
    #
    # slope: the slope of the negative side of the function
    # z_vectors: numpy array of z=wa+b vectors
    def __init__(self, slope=0.1):
        if slope == None:
            self.slope = 0.1
        else:
            self.slope = slope
    def magnitude(self, z_vectors):
        # return a layers activation values given its z-vectors
        return np.where(z_vectors<=0, self.slope*z_vectors, z_vectors)
    def derivative(self, z_vectors):
        # returns its derivative with respect to the z-vectors
        return np.where(z_vectors<=0, self.slope, 1)

class PRelu:
    # don't use this, its implementation is incomplete
    # the parametric relu activation function
    # leaky relu but the shallow slope is learned during gradient descent
    #
    # slope: the slope of the negative side of the function
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors, slope):
        # return a layers activation values given its z-vectors
        return np.where(z_vectors<=0, slope*z_vectors, z_vectors)
    @staticmethod
    def derivative(z_vectors, slope):
        # returns its derivative with respect to the z-vectors
        return np.where(z_vectors<=0, slope, 1)

class Elu:
    # the elu activation function
    # similar to relu, but less suscepable to the dying relu problem
    # differentiable everywhere
    #
    # alpha: depth of the negative exponential portion
    # z_vectors: numpy array of z=wa+b vectors
    def __init__(self, alpha=1.0):
        if alpha == None:
            self.alpha = 1.0
        else:
            self.alpha = alpha
    def magnitude(self, z_vectors):
        # return a layers activation values given its z-vectors
        return np.where(z_vectors<=0, self.alpha*(np.exp(z_vectors)-1), z_vectors)
    def derivative(self, z_vectors):
        # returns its derivative with respect to the z-vectors
        return np.where(z_vectors<=0, self.magnitude(z_vectors)+self.alpha, 1)

class Softplus:
    # the softplus activation function
    # relu but differentiable everywhere
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return np.nan_to_num(np.log(1 + np.exp(z_vectors)))
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return 1.0 / (1.0 + np.exp(-z_vectors))

class Softmax:
    # the softmax activation function
    # basically a differentiable form of argmax
    #
    # z_vectors: numpy array of z=wa+b vectors
    @staticmethod
    def magnitude(z_vectors):
        # return a layers activation values given its z-vectors
        return np.exp(z_vectors) / np.sum(np.exp(z_vectors))
    @staticmethod
    def derivative(z_vectors):
        # returns its derivative with respect to the z-vectors
        return Softmax.magnitude(z_vectors) * (1 - Softmax.magnitude(z_vectors))

class MeanSquaredError:
    # the mean squared error cost function with L2 regularisation
    # use with relu like activation functions
    #
    # outputs: 2d numpy array of predictions made by the network
    # targets: 2d numpy array of correct prediction values
    # weights: list of numpy arrays of the networks weights. one array per layer
    # regularization_parameter: scalar L2 regularization parameter
    # training_dataset_size: scalar number of objects in the training dataset
    @staticmethod
    def magnitude(outputs, targets, weights, regularization_parameter, training_dataset_size):
        # returns the regularized magnitude of the cost function
        squared_weight_sum = 0
        for weights_array in weights:
            squared_weight_sum += np.sum(weights_array**2)
        regularization = (squared_weight_sum * regularization_parameter) / (2 * training_dataset_size)
        return (np.sum((outputs-targets)**2) / (2 * training_dataset_size) + regularization)
    @staticmethod
    def unregularized_derivative(outputs, targets):
        # returns an array of unregularized dcda values
        return outputs - targets

class BinaryCrossEntropy:
    # the binary cross entropy cost function with L2 regularisation
    # use with sigmoidal activation functions for binary classification
    #
    # outputs: 2d numpy array of predictions made by the network
    # targets: 2d numpy array of correct prediction values
    # weights: list of numpy arrays of the networks weights. one array per layer
    # regularization_parameter: scalar L2 regularization parameter
    # training_dataset_size: scalar number of objects in the training dataset
    @staticmethod
    def magnitude(outputs, targets, weights, regularization_parameter, training_dataset_size):
        #print("new one")
        #print(outputs)
        #print(targets)
        #print(weights)
        #print(regularization_parameter)
        #print(training_dataset_size)
        # returns the regularized magnitude of the cost function
        squared_weight_sum = 0
        for weights_array in weights:
            squared_weight_sum += np.sum(weights_array**2)
        regularization = (squared_weight_sum * regularization_parameter) / (2 * training_dataset_size)
        return -np.sum(np.nan_to_num(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs)) / training_dataset_size) + regularization
    @staticmethod
    def unregularized_derivative(outputs, targets):
        # returns an array of unregularized dcda values
        return (outputs - targets) / (outputs * (1 - outputs))

class CategoricalCrossEntropy:
    # the categorical cross entropy cost function with L2 regularisation
    # use with sigmoidal activation functions and the softmax output function
    #
    # outputs: 2d numpy array of predictions made by the network
    # targets: 2d numpy array of correct prediction values
    # weights: list of numpy arrays of the networks weights. one array per layer
    # regularization_parameter: scalar L2 regularization parameter
    # training_dataset_size: scalar number of objects in the training dataset
    @staticmethod
    def magnitude(outputs, targets, weights, regularization_parameter, training_dataset_size):
        # returns the regularized magnitude of the cost function
        squared_weight_sum = 0
        for weights_array in weights:
            squared_weight_sum += np.sum(weights_array**2)
        regularization = (squared_weight_sum * regularization_parameter) / (2 * training_dataset_size)
        return -(np.sum(np.nan_to_num(targets * np.log(outputs))) / training_dataset_size) + regularization
    @staticmethod
    def unregularized_derivative(outputs, targets):
        # returns an array of unregularized dcda values
        return outputs - targets

class Layer:
    # a layer class used by the NeuralNetwork class
    # don't directly call it, make a neural network with the NeuralNetwork class instead
    # not an input layer as that layer don't have any values
    #
    # number_inpus: scalar number of inputs to the layer
    # number_beurons: scalar number of neurons in the layer
    # activation_function: class title of the layers activation function
    # activation parameter: if the activation_function has a parameter, this is it. otherwise: None
    def __init__(self, number_inputs, number_neurons, activation_function, activation_parameter):
        self.counter = 0
        self.number_inputs = number_inputs
        self.number_neurons = number_neurons
        self.weights = np.random.normal(size = (number_neurons, number_inputs), scale = 2 / np.sqrt(number_inputs))
        self.biases = np.zeros(number_neurons)
        self.initialize_activation_function(activation_function, activation_parameter)
        self.inputs = np.asarray([])
        self.activations = np.asarray([])
        self.z_vectors = np.asarray([])

    def forward(self, inputs):
        # returns a numpy array of the layers activations
        #
        # inputs: numpy array of the previous layer's activations
        z_vectors = self.weights @ inputs + self.biases
        activations = self.activation_function.magnitude(z_vectors)
        self.inputs = inputs
        self.z_vectors = z_vectors
        self.activations = activations
        self.counter += 1
        return activations

    def backward(self, our_dcda, regularization_parameter, training_dataset_size):
        # returns numpy arrays of the derivatives of the layers weights and biases and the next layer's activations to the cost function
        #
        # our_dcda: numpy array of the derivative of the cost function with respect to this layers activations
        # regularization parameter: scalar parameter for the L2 regularization on the cost function
        # training_dataset_size: scalar number of features in the training dataset
        dcdz = our_dcda * self.activation_function.derivative(self.z_vectors)
        dcdb = dcdz
        dcdw = dcdz[:, np.newaxis] * self.inputs + regularization_parameter * self.weights / training_dataset_size
        next_dcda = self.weights.T @ dcdz
        return next_dcda, dcdb, dcdw

    def initialize_activation_function(self, activation_function, activation_parameter):
        # initialises the activation function
        #
        # activation_function: class title for the layers activation function
        # activation_parameter: scalar parameter for the activation function if it has one. otherwise: None
        if activation_function == LeakyRelu or activation_function == Elu:
            self.activation_function = activation_function(activation_parameter)
        else:
            self.activation_function = activation_function

    def __repr__(self):
        self_representation = (f"number_neurons: {self.number_neurons}\n" +
                              f"number_inputs: {self.number_inputs}\n" +
                              f"activation_function: {self.activation_function}\n" +
                              f"biases: {self.biases}\n" +
                              f"weights:\n{self.weights}\n" +
                              f"inputs: {self.inputs}\n" +
                              f"activations: {self.activations}\n" +
                              f"z_vectors: {self.z_vectors}")
        return self_representation

class NeuralNetwork:
    # a neural network class for doing cool stuff
    # use by instantiating it, then calling the SGD method to train it with stochastic gradient descent
    #
    # shape: list of the number of neurons per layer
    # cost: class title for the cost function
    # activation_function: class title for the activation function
    # activation_parameter: parameter for the activation function if there is one
    # output_function: class title for the activation function of the final layer if it's different from the other activation funciton
    def __init__(self, shape, cost=CategoricalCrossEntropy, activation_function=Sigmoid, activation_parameter=None, output_function=Softmax):
        self.shape = shape
        self.cost = cost
        self.activation_function = activation_function
        self.activation_parameter = activation_parameter
        self.output_function = output_function
        self.initialize_layers(shape, activation_function, activation_parameter, output_function)
        self.number_layers = len(self.layers) # doesn't count the input layer as it's not a layer object

    def initialize_layers(self, shape, activation_function, activation_parameter, output_function):
        # initialises the layers of the nural network
        #
        # shape: list of the number of neurons per layer
        # activation_function: class title for the activation function
        # activation_parameter: parameter for the activation function if there is one
        # output_function: class title for the activation function of the final layer if it's different from the other activation funciton
        layers = []
        counter = 0
        for layer_size_pair in pairwise(shape):
            if counter == len(shape)-2 and output_function!=None:
                layers.append(Layer(layer_size_pair[0], layer_size_pair[1], output_function, activation_parameter))
            else:
                layers.append(Layer(layer_size_pair[0], layer_size_pair[1], activation_function, activation_parameter))
            counter += 1
        self.layers = layers

    def feedforward(self, thought):
        # calculates the activations of the final layer given the activations of the input layer
        #
        # thought: current activations of the current layer
        for layer in self.layers:
            thought = layer.forward(thought)
        return thought

    def backpropogate(self, inputs, targets, regularization_parameter, training_dataset_size):
        # calculates the derivatives of the cost function to the weights and biases of the network for a single feature
        #
        # inputs: numpy array of a single feature of data
        # targets: numpy array of the corresponding label of data
        # regularization_parameter: scalar regularization parameter for the L2 normalisation
        # training_dataset_size: scalar count of features in the training dataset
        outputs = self.feedforward(inputs)
        final_weights = self.layers[-1].weights
        final_z_vectors = self.layers[-1].z_vectors
        final_inputs = self.layers[-1].z_vectors
        final_activation_function = self.layers[-1].activation_function
        next_dcda = self.cost.unregularized_derivative(outputs, targets)
        nabla_b = []
        nabla_w = []
        for layer in reversed(self.layers):
            next_dcda, dcdb, dcdw = layer.backward(next_dcda, regularization_parameter, training_dataset_size)
            nabla_b.append(dcdb)
            nabla_w.append(dcdw)
        return nabla_b[::-1], nabla_w[::-1]

    def SGD(self, train_features, train_labels, test_features, test_labels,
            epochs="automatic stopping", initial_learning_rate=0.02,
            regularization_parameter=1.0, printing=False, stopping_threshold=32,
            batch_size=300):
        # implements stochastic gradient descent
        #
        # train_features - test_labels: numpy arrays of features an labels
        # epochs: number of backpropogation steps. by default it attempts to stop when the network switches to overfitting
        # initial_learning_rate: initial learning rate of the network. will half when nearing optimum to allow for more fine grain optimisation
        # regularization_parameter: scalar regularization parameter for the L2 regularization
        # printing: boolean for whether or not to print out evaluation etrics while training
        # stopping_threshold: scalar determining when the optimisation will stop. smaller values stop on a lighter trigger
        # batch_size: scalar number of features per batch
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_costs = []
        self.test_costs = []
        self.minimum_test_cost = 1.0
        self.minimum_test_cost_epoch = 0
        self.stopping_threshold = stopping_threshold
        self.training_dataset_size = len(train_labels)
        self.learning_rate = initial_learning_rate
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.create_batches(batch_size)
        epoch = 0
        while self.check_for_stopping(epoch, epochs):
            self.scale_hyperparameters(epoch, epochs)
            for feature, label in zip(self.train_feature_batches[self.batch], self.train_label_batches[self.batch]):
                nabla_b, nabla_w = self.backpropogate(feature, label, regularization_parameter, self.training_dataset_size)
                for layer in range(self.number_layers):
                    self.layers[layer].weights -= self.learning_rate * nabla_w[layer]
                    self.layers[layer].biases -= self.learning_rate * nabla_b[layer]
            self.train_accuracies.append(self.evaluate_accuracy(train_features, train_labels))
            self.test_accuracies.append(self.evaluate_accuracy(test_features, test_labels))
            self.train_costs.append(self.evaluate_cost(train_features, train_labels, regularization_parameter))
            self.test_costs.append(self.evaluate_cost(test_features, test_labels, regularization_parameter))
            if self.test_costs[-1] < self.minimum_test_cost:
                self.minimum_test_cost = self.test_costs[-1]
                self.minimum_test_cost_epoch = epoch
            if printing:
                if epoch % 8 == 0:
                    print(f"epoch: {epoch}")
                    print(f"accuracy: {self.test_accuracies[-1]}")
                    print(f"cost {self.test_costs[-1]}")
            epoch += 1
            self.batch += 1
            if self.batch == self.batch_count:
                self.batch = 0
        self.evaluate_f1(test_features, test_labels)

    def create_batches(self, batch_size):
        # creates batches for use with stochastic gradient descent
        #
        # batch_size: scalar number of features per batch
        self.batch = 0
        self.batch_count = np.round(len(self.train_features) / batch_size)
        self.train_feature_batches = np.array_split(self.train_features, self.batch_count)
        self.train_label_batches = np.array_split(self.train_labels, self.batch_count)
        self.test_feature_batches = np.array_split(self.test_features, self.batch_count)
        self.test_label_batches = np.array_split(self.test_labels, self.batch_count)

    def scale_hyperparameters(self, epoch, epochs):
        # automatically rescales hyperparameters to allow for quicker learning in the beginning and more fine tuned optimisation in the end
        #
        # epoch: number of backpropogation cycles which have happened
        if epoch - self.minimum_test_cost_epoch > self.stopping_threshold / 2:
            self.learning_rate = self.learning_rate / 10
        elif type(epochs) == int and epoch == epochs - 64:
            self.learning_rate = self.learning_rate / 10

    def check_for_stopping(self, epoch, epochs):
        # returns True if the network has begun to overfit
        #
        # epoch: scalar number of backpropogation steps which have happened
        # epochs: scalar number of epochs to run if the user doesn't want automated stopping
        if epoch > 1024: # hard upper limit
            return False
        elif type(epochs) == int:
            return epoch <= epochs
        elif epoch <= self.stopping_threshold:
            return True
        elif epochs == "automatic stopping":
            return epoch - self.minimum_test_cost_epoch <= self.stopping_threshold
        return False

    def evaluate_accuracy(self, inputs, targets):
        # evaluates the accuray of the network on a dataset
        #
        # inputs: numpy array of data features
        # targets: numpy array of data labels
        answers = np.array([np.argmax(self.feedforward(aninput)) for aninput in inputs])
        labels = np.array([np.argmax(target) for target in targets])
        return np.sum(answers == labels) / len(targets)

    def evaluate_cost(self, inputs, targets, regularization_parameter):
        # evaluates the cost of the network on a dataset
        #
        # inputs: numpy array of data features
        # targets: numpy array of data labels
        # regularization_parameter: regularization_parameter for the L2 regularization
        outputs = np.array([self.feedforward(aninput) for aninput in inputs])
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
        return self.cost.magnitude(outputs, targets, weights, regularization_parameter, len(inputs))

    def evaluate_f1(self, inputs, targets):
        # evaluates the f1 score of the network on a dataset
        # the positive case is defined to be the left column
        #
        # inputs: numpy array of data features
        # targets: numpy array of data labels
        intermediate_answers = np.array([np.argmax(self.feedforward(aninput)) for aninput in inputs])
        answers = np.zeros(targets.shape).tolist()
        counter = 0
        for answer in answers:
            answer[intermediate_answers[counter]] = 1
            counter += 1
        answers = np.asarray(answers)
        self.true_positives = np.sum(targets[x][0] == 1 and answers[x][0] == 1 for x in range(len(inputs)))
        self.false_positives = np.sum(targets[x][0] == 0 and answers[x][0] == 1 for x in range(len(inputs)))
        self.true_negatives = np.sum(targets[x][0] == 0 and answers[x][0] == 0 for x in range(len(inputs)))
        self.false_negatives = np.sum(targets[x][0] == 1 and answers[x][0] == 0 for x in range(len(inputs)))
        self.f1_score = (2 * self.true_positives) / (2 * self.true_positives + self.false_positives + self.false_negatives)

    def plot_performance(self):
        # plots accuracies and costs for the network during training
        # and prints out a bunch of evaluation metrics
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(self.train_accuracies, label='training accuracies')
        ax[0].plot(self.test_accuracies, label='test accuracies')
        ax[0].legend()
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy');
        ax[1].plot(self.train_costs, label='training costs')
        ax[1].plot(self.test_costs, label='test costs')
        ax[1].legend()
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Cost');
        print(f"f1 score: {self.f1_score}")
        print(f"true positives: {self.true_positives}")
        print(f"true negatives: {self.true_negatives}")
        print(f"false positives: {self.false_positives}")
        print(f"false negatives: {self.false_negatives}")
        print(f"final accuracy: {self.test_accuracies[-1]}")
        print(f"final cost: {self.test_costs[-1]}")
    
    def __repr__(self):
        self_description = f" shape:{self.shape}\n cost: {self.cost}\n"
        counter = 1
        for layer in self.layers:
            self_description += f"\n layer {counter}\n{repr(layer)}"
            counter += 1
        return self_description

def test_layer():
    print("test_layer()")
    # If successful, these tests should not raise an exception.
    # NOTE - they are picky about the naming of attributes!
    layer = Layer(4, 6, activation_function=Sigmoid, activation_parameter=None)
    nt.assert_equal(layer.weights.shape, (6, 4))
    # set layer weights to unity matrix
    layer.weights = np.ones(shape=layer.weights.shape)
    # set layer biasses to layer number
    layer.biases = np.arange(layer.biases.size)
    # These tests check the Layer.forward function.
    x = np.array([-0.27,  0.45,  0.64, 0.31])
    # feed forward
    output = layer.forward(x)
    # check we stored the properties
    for attr in ('activations', 'z_vectors', 'inputs'):
        if not hasattr(layer, attr):
            msg = f"Layer does not have {attr} attribute. Perhaps you used a different name?"
            raise AssertionError(msg)
    # check the values
    np.testing.assert_allclose(output, layer.activations)
    np.testing.assert_allclose(layer.z_vectors, layer.weights @ x + layer.biases)
    np.testing.assert_allclose(layer.inputs, x)
    # test layer.backwards()
    dC_da = np.ones_like(layer.activations)
    dC_da, dC_db, dC_dw = layer.backward(dC_da, 0, 1)
    # check shapes
    nt.assert_equal(dC_db.shape, layer.biases.shape)
    nt.assert_equal(dC_dw.shape, layer.weights.shape)
    nt.assert_equal(dC_da.shape, x.shape)
    # check values
    np.testing.assert_allclose(
        dC_da, 
        np.array([0.34320402, 0.34320402, 0.34320402, 0.34320402]),
        atol=1.0e-8
    )
    np.testing.assert_allclose(
        dC_db, 
        np.array([0.18454646, 0.09493337, 0.04013212, 
              0.01557778, 0.00584717, 0.00216714]),
        atol=1.0e-8
    )
    np.testing.assert_allclose(
        dC_dw,
        np.array(
            [[-0.04982754,  0.08304591,  0.11810973,  0.0572094 ],
             [-0.02563201,  0.04272002,  0.06075736,  0.02942934],
             [-0.01083567,  0.01805945,  0.02568456,  0.01244096],
             [-0.004206  ,  0.00701   ,  0.00996978,  0.00482911],
             [-0.00157873,  0.00263122,  0.00374219,  0.00181262],
             [-0.00058513,  0.00097521,  0.00138697,  0.00067181]]),
        atol=1.0e-8
    )
    print("passed")

def test_neural_network():
    print("test_neural_network()")
    net = NeuralNetwork([4, 5, 5, 1], activation_function=Sigmoid, cost=MeanSquaredError, output_function=None)
    # load in pre-saved weights and biasses for this network
    # (this network classifies galaxies)
    from src.network_tools import load_network_params
    # loads pre-trained weights and biasses for a [4, 5, 5, 1] sized     network. Will work if NeuralNetwork class meets specs
    net = load_network_params(net, 'data/trained_qso_weights.npz',         'data/trained_qso_biasses.npz')
    # test example - a  normal galaxy
    x = np.array([1.99211, 1.0415 , 0.55032, 0.36946])
    y = 0  # expected class
    nabla_b, nabla_w = net.backpropogate(x, y, 0, 1)
    # check entries in lists are the correct shape
    for l in range(net.number_layers):
        layer = net.layers[l]
        if layer.biases.shape != nabla_b[l].shape:
            raise RuntimeError(f'bias shape mismatch for {l}')
        if layer.weights.shape != nabla_w[l].shape:
            raise RuntimeError(f'weight shape mismatch for {l}')
    # check numerical values in first hidden layer 
    nabla_b_1 = nabla_b[0]
    err_msg = """
    numerical values in gradient w.r.t bias looks wrong.
    make sure you are calculating this part right, and that 
    you are returning nabla_b, nabla_w in that order.
    """
    nt.assert_almost_equal(np.sum(nabla_b_1), -5.5775191560733e-06,     msg=err_msg)
    nabla_w_1 = nabla_w[0]
    err_msg = """
    numerical values in gradient w.r.t weights looks wrong.
    make sure you are calculating this part right, and that 
    you are returning nabla_b, nabla_w in that order.
    """
    nt.assert_almost_equal(np.sum(nabla_w_1), -2.2050108456428612e-05, msg=err_msg)
    print("passed")

def test_neural_network_initialization():
    network = NeuralNetwork([204,13,75,76], cost=CategoricalCrossEntropy, activation_function=LeakyRelu,
                 activation_parameter=34, output_function=Softmax)
    assert network.layers[0].number_inputs == 204
    assert network.layers[0].number_neurons == 13
    assert network.layers[1].number_inputs == 13
    assert network.layers[1].number_neurons == 75
    assert network.layers[2].number_inputs == 75
    assert network.layers[2].number_neurons == 76
    assert type(network.layers[1].activation_function) == LeakyRelu
    assert network.layers[1].activation_function.slope == 34
    assert network.layers[2].activation_function == Softmax

def test_training():
    print("test_training()")
    digits = load_digits()
    lb = preprocessing.LabelBinarizer()
    lb.fit(digits.target)
    y = lb.transform(digits.target)
    #print(y.shape)
    #print(f"{digits.target[6]} -> {y[6]}")
    X = digits.data
    train_features, test_features, train_labels, test_labels = train_test_split(X, y)
    network = NeuralNetwork([64,30,10])
    network.SGD(train_features, train_labels, test_features, test_labels, printing=True)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(network.train_accuracies, label='training accuracies')
    ax[0].plot(network.test_accuracies, label='test accuracies')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy');
    ax[1].plot(network.train_costs, label='training costs')
    ax[1].plot(network.test_costs, label='test costs')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Cost');
    print("passed")

def test_functions():
    # tests activation, cost, and output functions
    print("test_functions()")
    z_vectors = np.asarray([1.0,-0.2])
    alpha = 0.1
    outputs = np.asarray([[0.4,0.9],[0.7,0.1]])
    targets = np.asarray([[0.5,0.7],[0.1,0.101]])
    regularization_parameter = 1.2
    training_dataset_size = 1356
    weights = [np.asarray([1.2,2.2]),np.asarray([3.4])]
    number_inputs = 2048
    number_neurons = 1024
    # sigmoid
    sigmoid_magnitudes = np.asarray([0.73105857863, 0.450166002688])
    sigmoid_derivatives = np.asarray([0.196611933241, 0.247516572712])
    sigmoid = Sigmoid()
    np.testing.assert_allclose(sigmoid.magnitude(z_vectors), sigmoid_magnitudes)
    np.testing.assert_allclose(sigmoid.derivative(z_vectors), sigmoid_derivatives)
    # tanh
    tanh_magnitudes = np.asarray([0.761594155956, -0.197375320225])
    tanh_derivatives = np.asarray([0.419974341614, 0.961042982966])
    tanh = Tanh()
    np.testing.assert_allclose(tanh.magnitude(z_vectors), tanh_magnitudes)
    np.testing.assert_allclose(tanh.derivative(z_vectors), tanh_derivatives)
    # arctan
    arctan_magnitudes = np.asarray([0.785398163397, -0.19739555985])
    arctan_derivatives = np.asarray([0.5, 0.961538461538])
    arctan = Arctan()
    np.testing.assert_allclose(arctan.magnitude(z_vectors), arctan_magnitudes)
    np.testing.assert_allclose(arctan.derivative(z_vectors), arctan_derivatives)
    # relu
    relu_magnitudes = np.asarray([1.0, 0.0])
    relu_derivatives = np.asarray([1.0, 0.0])
    relu = Relu()
    np.testing.assert_allclose(relu.magnitude(z_vectors), relu_magnitudes)
    np.testing.assert_allclose(relu.derivative(z_vectors), relu_derivatives)
    # leaky relu
    leaky_relu_magnitudes = np.asarray([1.0, -0.02])
    leaky_relu_derivatives = np.asarray([1.0, 0.1])
    leaky_relu = LeakyRelu()
    np.testing.assert_allclose(leaky_relu.magnitude(z_vectors), leaky_relu_magnitudes)
    np.testing.assert_allclose(leaky_relu.derivative(z_vectors), leaky_relu_derivatives)
    # parametric relu
    parameteric_relu_magnitudes = np.asarray([1.0, -0.02])
    parameteric_relu_derivatives = np.asarray([1.0, 0.1])
    parameteric_relu = PRelu()
    np.testing.assert_allclose(parameteric_relu.magnitude(z_vectors, alpha), parameteric_relu_magnitudes)
    np.testing.assert_allclose(parameteric_relu.derivative(z_vectors, alpha), parameteric_relu_derivatives)
    # elu
    elu_magnitudes = np.asarray([1.0, -0.181269246922])
    elu_derivatives = np.asarray([1.0, 0.818730753078])
    elu = Elu()
    np.testing.assert_allclose(elu.magnitude(z_vectors), elu_magnitudes)
    np.testing.assert_allclose(elu.derivative(z_vectors), elu_derivatives)
    # softplus
    softplus_magnitudes = np.asarray([1.31326168752, 0.598138869382])
    softplus_derivatives = np.asarray([0.73105857863, 0.450166002688])
    softplus = Softplus()
    np.testing.assert_allclose(softplus.magnitude(z_vectors), softplus_magnitudes)
    np.testing.assert_allclose(softplus.derivative(z_vectors), softplus_derivatives)
    # softmax
    softmax_magnitudes = np.asarray([0.768524783499, 0.231475216501])
    softmax_derivatives = np.asarray([0.177894440647, 0.177894440647])
    softmax = Softmax()
    np.testing.assert_allclose(softmax.magnitude(z_vectors), softmax_magnitudes)
    np.testing.assert_allclose(softmax.derivative(z_vectors), softmax_derivatives)
    # mean squared error
    mean_squared_error_magnitude = 0.00804498561947
    mean_squared_error_derivatives = np.asarray([[-0.1, 0.2],[0.6,-0.001]])
    mean_squared_error = MeanSquaredError()
    np.testing.assert_allclose(mean_squared_error.magnitude(outputs, targets, weights, regularization_parameter,
                                                            training_dataset_size), mean_squared_error_magnitude)
    np.testing.assert_allclose(mean_squared_error.unregularized_derivative(outputs, targets), mean_squared_error_derivatives)
    # binary cross entropy
    binary_cross_entropy_magnitude = 0.0100505968163
    binary_cross_entropy_derivatives = np.asarray([[-0.416666666667, 2.22222222222],
                                                   [2.85714285714,-0.0111111111111]])
    binary_cross_entropy = BinaryCrossEntropy()
    np.testing.assert_allclose(binary_cross_entropy.magnitude(outputs, targets, weights, regularization_parameter,
                                                              training_dataset_size), binary_cross_entropy_magnitude)
    np.testing.assert_allclose(binary_cross_entropy.unregularized_derivative(outputs, targets), binary_cross_entropy_derivatives)
    # categorical cross entropy
    categorical_cross_entropy_magnitude = 0.00848386896437
    categorical_cross_entropy_derivatives = np.asarray([[-0.1, 0.2],[0.6,-0.001]])
    categorical_cross_entropy = CategoricalCrossEntropy()
    np.testing.assert_allclose(categorical_cross_entropy.magnitude(outputs, targets, weights, regularization_parameter,
                                                               training_dataset_size), categorical_cross_entropy_magnitude)
    np.testing.assert_allclose(categorical_cross_entropy.unregularized_derivative(outputs, targets), categorical_cross_entropy_derivatives)
    print("passed")

def test_everything():
    test_layer()
    test_neural_network()
    test_neural_network_initialization()
    test_functions()
    test_training()
    print("all passed")







































































