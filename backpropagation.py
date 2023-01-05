import numpy as np 
import forward_propagation as fp 
import getParameters as param

class Backpropagation():

    def __init__(self, input_matrix, target_matrix, weight, bias, activation_function = 'sigmoid', learning_rate = 0.001) -> None:
        self.weight = weight
        self.bias = bias
        self.input_matrix = np.array(input_matrix)
        self.target_matrix = np.array(target_matrix)
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        if self.input_matrix.shape[0] != self.target_matrix.shape[0]:
            print("Error [Dimension mismatch]: I/P matrix of shape ({}, ?), Target matrix of shape ({}, ?)".format(self.input_matrix.shape[0],self.target_matrix.shape[0]))
            exit()
        # self.get_parameters()
        self.backpropagate()
    
    def return_parameters(self):
        return self.weight, self.bias
    
    def backpropagate(self):
        for i, inp_row in enumerate(self.input_matrix):
            BCKP = Backpropagation_internal(self.weight, self.bias, inp_row, self.target_matrix[i], 'sigmoid', self.learning_rate)
            self.weight, self.bias = BCKP.new_parameters()
    
    


class Backpropagation_internal:
    def __init__(self, weight, bias, input_, target_, activation_function = 'sigmoid', learning_rate = 0.001) -> None:
        self.weight = weight
        self.bias = bias
        self.input_ = np.array(input_)
        self.target_ = np.array(target_)
        self.activation_function = activation_function
        self.layer_out = None
        self.learning_rate = learning_rate
        self.forward_propagation()
    
    def forward_propagation(self):
        FP = fp.Forward_Propagation(self.weight, self.bias, self.input_, self.activation_function)
        _ = FP.forward_propagation()
        self.layer_out = FP.layer_output()
        self.backpropagation()
    
    def activation_differential(self, IP):
        if self.activation_function == 'sigmoid':
            return (1 - IP)*IP
        # elif self.activation_function == 'relu' or self.activation_function == 'step':
        #     for i in range(len(IP)):
        #         if IP[i] != 0:
        #             IP[i] = 1 
    
    def new_parameters(self):
        return self.weight, self.bias
    
    def backpropagation(self):
        # error is the target - input | all the calcuation are based on Sum of squared residual
        error_ = self.target_ - self.layer_out[len(self.layer_out) - 1]
        for i in range(len(self.layer_out) - 1, 0, -1):
            # going from the back considering 2 layers from the last (or the last interface)
            IP = self.layer_out[i]
            OP = self.layer_out[i - 1]
            M1 = -error_ *(self.activation_differential(IP))
            OP = OP.reshape((len(OP), 1))
            M1 = M1.reshape((1, len(M1)))
            
            # dError/dWeight = - Error *(A')*(OP of the previous layer)
            # for A = sigmoid : dError/dWeight = - Error *(1 - IP)(IP)*(OP of the previous layer)
            dError_dWeight = np.matmul(OP, M1)
            interface = i - 1
            
            # dError/dBias = - Error *(A')
            # for A = sigmoid : dError/dBias = - Error *(1 - IP)(IP)
            dError_dBias = M1
            
            # write for the previous interface (or next iteration) error
            new_error_ = np.zeros(len(self.layer_out[i - 1]))
            sum_array = sum(self.weight[interface])
            for a, row in enumerate(self.weight[interface]):
                for b, wt in enumerate(row):
                    new_error_[a] += (wt/sum_array[b])*error_[b]
            error_ = new_error_
            # print(error_)

            # W = W - LR*dErrot/dWeight
            # interface = i - 1
            self.weight[interface] -= self.learning_rate*dError_dWeight

            # B = B - LR*dError/dWeight
            dError_dBias = dError_dBias.reshape(len(dError_dBias[0]))
            self.bias[i] -= self.learning_rate*dError_dBias

