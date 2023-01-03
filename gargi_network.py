'''
    API for the GARGi Networks
'''

import neural_architecture as na
import getParameters as param
import forward_propagation as fp

def build_network(architecture):
    neural_arch = na.Neural_Architecture(architecture = architecture)

def get_parameters():
    params = param.GetParameters()
    weight, bias = params.getParameters()
    return weight, bias

def forward_propagation(weight, bias, input_, activation_func = 'sigmoid'):
    FP = fp.Forward_Propagation(weight, bias, input_, activation_function = activation_func)
    OP = FP.forward_propagation()
    layer_out = FP.layer_output()
    return OP, layer_out


    
