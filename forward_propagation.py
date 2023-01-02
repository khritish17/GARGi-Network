import numpy as np 

class Forward_Propagation:
    def __init__(self, weight, bias, input_, activation_function = 'sigmoid') -> None:
        self.weight = weight
        self.bias = bias
        self.input_ = input_
        self.activation_function = activation_function
        self.interface_out = {}
        # self.forward_propagation()

    def weighted_input(self, input_, weight_matrix):
        # simply perform the matrix multiplication
        return np.matmul(input_, weight_matrix)
    
    def activation(self, input_, activation_function):
        # here the input is just a float value and not an array or list
        if activation_function == 'sigmoid':
            return 1/(1 + np.exp(-input_))
        elif activation_function == 'relu':
            return max(0, input_)
        elif activation_function == 'step':
            if input_ > 0:
                return 1
            else:
                return 0

    def bias_activation(self, bias, input_):
        # we will receive input_ and the bias as a vector (same dimension)
        for i in range(len(input_)):
            input_[i] = self.activation(input_[i] + bias[i], activation_function=self.activation_function) 
        return input_
    
    def interface_output(self):
        pass

    def forward_propagation(self):
        OP = self.input_
        # perform the weighted input 
        for interface, weight in self.weight.items():
            OP = self.weighted_input(OP, weight)
            # OP = self.bias[interface + 1] + OP
            OP = self.bias_activation(self.bias[interface + 1], OP)
        # print(OP)
        return OP
