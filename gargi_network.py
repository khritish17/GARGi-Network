'''
    API for the GARGi Networks
'''

import neural_architecture as na
# import getParameters as param
# import forward_propagation as fp
# import forward_propagation2 as fp2
def build_network(architecture):
    neural_arch = na.Neural_Architecture(architecture = architecture)

# def get_parameters():
#     params = param.GetParameters()
#     weight, bias = params.getParameters()
#     return weight, bias

# # def forward_propagation(weight, bias, input_, activation_func = 'sigmoid'):
# #     FP = fp.Forward_Propagation(weight, bias, input_, activation_function = activation_func)
# #     OP = FP.forward_propagation()
# #     return OP


# def forward_propagation2(weight, bias, input_, activation_func = 'sigmoid'):
#     FP = fp2.Forward_Propagation(weight, bias, input_, activation_function = activation_func)
#     OP = FP.forward_propagation()
#     return OP
    
