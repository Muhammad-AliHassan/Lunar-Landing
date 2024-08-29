import random
import math

learning_rate = 0.8

class Neuron:
    def __init__(self, layer_index, activation_value, num_weights):
        self.activation = activation_value
        self.delta_weights = []
        self.gradient = 0
        self.weights = [random.random() for _ in range(num_weights)]
        self.initialize_delta_weights(num_weights)
        self.index = layer_index 

    def initialize_delta_weights(self, num_weights):
        self.delta_weights = [0] * num_weights

    def calculate_weights(self, layer):
        weighted_sum = sum(float(neuron.activation) * float(neuron.weights[self.index]) for neuron in layer)
        self.activation = self.apply_sigmoid(weighted_sum)

    @staticmethod
    def apply_sigmoid(x):
        return 1 / (1 + math.exp(-x))
