import gargi_network as nn
import backpropagation as bckp
# nn.build_network([2, 3, 2])
w, b = nn.get_parameters()
print("Weights")
print(w)
# print("Bias")
# print(b)
op, layer_out = nn.forward_propagation(w, b, [1, 2], 'sigmoid')

# print(layer_out)
# print(op)
BCKP = bckp.Backpropagation(w, b, [1, 2], [3, 4], 'sigmoid')
wt, bi = BCKP.new_parameters()
print("New Weights")
print(wt)
# print("Bias")
# print(b)