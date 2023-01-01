import csv  
import numpy as np

class GetParameters:
    def __init__(self):
        '''
            It assumes the weights are avialable in weights.csv
        '''
        self.index = {}
        self.weights = {}
        self.bias = {}
        
        
    
    def getParameters(self):      
        self.getIndex()
        self.getWeights()
        self.getBias()
        return self.weights, self.bias

    def getIndex(self):
        index_file = open('index_weights.csv', mode = 'r')
        index_csv = csv.reader(index_file)
        for line in index_csv:
            self.index[line[0]] = line[1:3]
        del self.index['layer_interface']
        # print(self.index)
        index_file.close()

    def getWeights(self):
        weight_file = open('weights.csv', mode='r')
        weight_csv = csv.reader(weight_file)
        for interface, ranges in self.index.items():
            begin_, end_ = int(ranges[0]), int(ranges[1])
            weight_matrix = []
            for i in range(begin_, end_ + 1):
                for line in weight_csv:
                    weight_matrix.append([float(l) for l in line]) 
                    break
            self.weights[int(interface)] = np.array(weight_matrix)
        # print(self.weights[1])
        weight_file.close()
    
    def getBias(self):
        bias_file = open('bias.csv', mode = 'r')
        bias_csv = csv.reader(bias_file)
        count = 0
        for line in bias_csv:
            self.bias[count] = np.array([float(l) for l in line])
            count += 1
        bias_file.close()

