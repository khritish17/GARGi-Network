# Essential packages
'''
    ' neural_architecture.py ' generates the neural architecture 
    i.e. Input nodes, Output nodes, Hidden layers and Hidden Nodes 

    should be called only once at the begining, do not call after training
    otherwise optimized weights will be re-written 
'''
import numpy as np

class Neural_Architecture:
    def __init__(self, architecture) -> None:
        print('Neural architecture build initiated\n')
        self.architecture = architecture
        self.weight_file_name = 'weights'
        self.bias_file_name = 'bias'
        self.weights = {}
        self.bias = {}
        self.cache_weights()
        self.cache_bias()
        print('Neural architecture build complete\n')
    
    def cache_weights(self):
        print('\tWeights caching initiated\n\t[STANDBY]')
        for i in range(len(self.architecture) - 1):
            layer_i = self.architecture[i]
            layer_inext = self.architecture[i + 1]
            self.weights[i] = np.random.randn(layer_i, layer_inext)
        # print(self.weights)
        print('\tWeights caching successfully completed _/\n')
        self.save_weights()
    
    def save_weights(self):
        weights_file = open('{}.csv'.format(self.weight_file_name),'w')
        index_weights_file = open('{}.csv'.format('index_weights'),'w')
        index_weights_file.write('{},{},{},{},{}\n'.format('layer_interface', 'row begin', 'row end', 'column begin', 'column end'))
        print('\tThe {}.csv file has been successfully created _/\n'.format(self.weight_file_name))
        print('\tWeights persistent saving initiated\n\t[STANDBY]')
        
        line_count = 0
        matrix_count = 0
        for val in self.weights.values():
            # the val is 2d matrix of weights of 2 adjacent layers 
            m = len(val)
            n = len(val[0])
            index_weights_file.write('{},{},{},{},{}\n'.format(matrix_count, line_count, line_count + m - 1, 0, n - 1))
            line_count += m
            matrix_count += 1
            for i in range(m):
                for j in range(n):
                    if j == n - 1:
                        weights_file.write('{}\n'.format(val[i][j]))
                    else:
                        weights_file.write('{},'.format(val[i][j]))
        
        
        weights_file.close()
        index_weights_file.close()
        print('\tWeights persistent saving successfully completed _/')
    
    def cache_bias(self):
        print('\tBias caching initiated\n\t[STANDBY]')
        for i, node in enumerate(self.architecture):
            if i == 0:
                self.bias[i] = np.zeros((node, 1))
            else:
                self.bias[i] = np.random.randn(node, 1)
        print('\tBias caching successfully completed _/\n')
        # print(self.bias)
        self.save_bias()
    
    def save_bias(self):
        bias_file = open('{}.csv'.format(self.bias_file_name),'w')
        
        print('\tThe {}.csv file has been successfully created _/\n'.format(self.bias_file_name))
        print('\tBias persistent saving initiated\n\t[STANDBY]')
        for val in self.bias.values():
            for i,b in enumerate(val):
                if i == len(val) - 1:
                    bias_file.write('{}\n'.format(b[0]))
                else:
                    bias_file.write('{},'.format(b[0]))
        bias_file.close()
        
        print('\tBias persistent saving successfully completed _/')




        

    def help(self) -> None:
        print('Helper Menu:')
        print('[1] Input Type: [ List<int> ]')
        print('[2] Each entry in the list represents a layer and the value represents the no. of neurons in that layer')
        print('[3] First entry: Input Layer, Last entry: Output Layer')
        print('[4] Other remaining entries represents Hidden Layer(s)')
        print('e.g.')
        print('For the Neural Architecture: 3 Input_Neurons in the Input layer')
        print('                             2 Hidden_Layers, one layer with 3 neurons and another with 2 neurons')
        print('                             4 Output_Neurons in the Output layer')  
        print('Input_List should be: [3, 2, 4]')

# NA = Neural_Architecture([3 , 2 , 5 , 2 , 4 , 4])

# NA.help()