import gargi_network as nn
import backpropagation as bckp
# nn.build_network([2, 3, 2])
w, b = nn.get_parameters()
# print("Weights")
# print(w)
print("Bias")
print(b)
op, layer_out = nn.forward_propagation(w, b, [1, 2], 'sigmoid')

# print(layer_out)
# print(op)
BCKP = bckp.Backpropagation([[1, 2], [3, 4], [5, 6]], [[6, 1], [7, 0], [4, 3]], w, b)

wt, bi = BCKP.return_parameters()
# print("New Weights")
# print(wt)
print("New Bias")
print(bi)