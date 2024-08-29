from Neuron import Neuron
import math
from csv import reader

iterations = 1000
global inv_scaling
inv_scaling = False


input_layer = []
hidden_layer = []
output_layer = []


lambda_neural = 0.8
learning_rate = 0.8
momentum_neural = 0.5

training = 0

def layer_initialize(input_neurons, hidden_neurons, output_neurons ):
    global input_layer, hidden_layer, output_layer 
    input_layer = [Neuron(i, 0, 2) for i in range(input_neurons + 1)]
    hidden_layer = [Neuron(i, 0, 2) for i in range(hidden_neurons + 1)]
    output_layer = [Neuron(i, 0, 0) for i in range(output_neurons)]


def forward_propagation(input_row):
    for i in range(len(input_layer)):
        input_layer[i].activation = input_row[i]
    for i in range(len(hidden_layer) - 1):
        hidden_layer[i].calculate_weights(input_layer)
    hidden_layer[-1].activation = 1
    for i in range(len(output_layer)):
        output_layer[i].calculate_weights(hidden_layer)


def backward_propagation(outputs_network):
    errors = []
    for i in range(len(output_layer)):
        errors.append(float(outputs_network[i]) - float(output_layer[i].activation))

    for i in range(len(output_layer)):
        output_layer[i].gradient = lambda_neural * output_layer[i].activation * (1 - output_layer[i].activation) * errors[i]

    for i in range(len(hidden_layer)):
        result = sum(float(output_layer[j].gradient) * float(hidden_layer[i].weights[j]) for j in range(len(output_layer)))
        hidden_layer[i].gradient = lambda_neural * hidden_layer[i].activation * (1 - hidden_layer[i].activation) * result

    update_weights(hidden_layer, output_layer)
    update_weights(input_layer, hidden_layer)

def update_weights(from_layer, to_layer):
    for i in range(len(from_layer)):
        for j in range(len(to_layer) - 1):
            from_layer[i].delta_weights[j] = learning_rate * float(to_layer[j].gradient) * float(from_layer[i].activation) + momentum_neural * float(from_layer[i].delta_weights[j])

    for i in range(len(from_layer)):
        from_layer[i].weights = [float(from_layer[i].weights[j]) + float(from_layer[i].delta_weights[j]) for j in range(len(from_layer[i].weights))]


def calculate_average_epoch_error():
    errors = []
    with open('training.csv', 'r') as training_data_file:
        data_reader = reader(training_data_file)
        for row in data_reader:
            forward_propagation([row[0], row[1], 1])
            error = calculate_squared_error(row[2], output_layer[0].activation, row[3], output_layer[1].activation)
            errors.append(error)
    return math.sqrt(sum(errors) / len(errors))

def calculate_average_validation_error():
    errors = []
    with open('testing.csv', 'r') as validating_data_file:
        data_reader = reader(validating_data_file)
        for row in data_reader:
            forward_propagation([row[0], row[1], 1])
            error = calculate_squared_error(row[2], output_layer[0].activation, row[3], output_layer[1].activation)
            errors.append(error)
    return math.sqrt(sum(errors) / len(errors))


def calculate_squared_error(target1, output1, target2, output2):
    return ((float(target1) - float(output1)) ** 2 + (float(target2) - float(output2)) ** 2) / 2


def process_and_predict(row_data, x1_max, x2_max, y1_max, y2_max, x1_min, x2_min, y1_min, y2_min):
    layer_initialize(input_neurons, hidden_neurons, output_neurons )
    with open('weights_lunar_landar.txt') as file:
        weights = file.read().split(",")

    index_flag = 0

    for i in range(len(input_layer)):
        for j in range(len(input_layer[i].weights)):
            input_layer[i].weights[j] = weights[index_flag]
            index_flag += 1

    inv_scaling = True if row_data[0] > 0 else False

    for i in range(len(hidden_layer)):
        for j in range(len(hidden_layer[i].weights)):
            hidden_layer[i].weights[j] = weights[index_flag]
            index_flag += 1

    scaled_input_one = (row_data[0] - x1_min) / (x1_max - x1_min)
    scaled_input_two = (row_data[1] - x2_min) / (x2_max - x2_min)
    scaled_result = [scaled_input_one, scaled_input_two, 1]

    forward_propagation(scaled_result)

    unscaled_output_one = output_layer[0].activation * (y1_max - y1_min) + y1_min
    unscaled_output_two = output_layer[1].activation * (y2_max - y2_min) + y2_min

    output_one = abs(unscaled_output_one) if inv_scaling else -abs(unscaled_output_two)
    output_two = abs(unscaled_output_two)

    return [output_one, output_two]


def training_lunar():
    for epoch in range(iterations):
        with open('training.csv', 'r') as training_file:
            data = reader(training_file)
            for row in data:
                forward_propagation([row[0], row[1], 1])
                backward_propagation([row[2], row[3]])

        training_error = calculate_average_epoch_error()
        testing_error = calculate_average_validation_error()
        print("****************************************")
        print('Epoch:', epoch + 1, 'Training Error:', training_error, 'Testing Error', testing_error)

    with open("weights_lunar_lander.txt", "w") as file:
        for layer in [input_layer, hidden_layer]:
            for neuron in layer:
                for weight in neuron.weights:
                    file.write(str(weight) + ",")

    print('Training Ended')



input_neurons = 2
hidden_neurons = 2
output_neurons = 2

if(training == 1):
    layer_initialize(input_neurons, hidden_neurons, output_neurons )
    print(input_layer)
    print(hidden_layer)
    print(output_layer)
    training_lunar()
    