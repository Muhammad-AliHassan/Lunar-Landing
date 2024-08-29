from LL_Neural_Network import process_and_predict

x1_max = 556.311
x2_max = 558.184
y1_max = 7.634
y2_max = 3.105
x1_min = -539.680
x2_min = 65.201
y1_min = -3.624
y2_min = -3.030

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    def predict(self, input_row):
        input = input_row.split(",") 
        output = process_and_predict([float(input[0]), float(input[1])], x1_max, x2_max, y1_max, y2_max, x1_min, x2_min, y1_min, y2_min)
        return output