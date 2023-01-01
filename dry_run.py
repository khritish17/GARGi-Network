import gargi_network as nn
# nn.build_network([2, 3, 2])
w, b = nn.get_parameters()
print("Weights")
print(w)
print("Bias")
print(b)
# op = nn.forward_propagation2(w, b, [1, 2], 'sigmoid')
# print(op)
