# GARGI-Network
Official Documentation of the GARGI Network

> **Note** The project is currently active, there may be some updates where the old API may be changed according to the requirement. Hence always look out for updates. Once the project gets finalized this NOTE will be removed.

### Description:

A simple artificial neural network from scratch using Python. 

### Prerequisites

- Install the latest version of Python, make sure to install the PIP pipeline 
- Install Numpy library

### Installation

- Clone the project from [GITLINK](https://github.com/khritish17/GARGi-Network.git)

### Contact

Author: Khritish Kumar Behera

Email: 
<khritish.official[at]gmail.com>


# GARGi Networks API

## build_network(*architecture*) | return type: None

Creates the neural network according to the specifications provided by the architecture. It generates the weight matrix and bias vectors. The weight matrix and the bias vector are saved in the weights.csv and bias.csv . 

The architecture is a list of no. of nodes at each layer starting from the input layer     (0 index) to the output layer ( n - 1 index); where n is the length of the architecture 

**e.g.**

    architecture  = [3, 2, 4, 5]
> **Note**
> - Input layer: 		3 nodes
> - 1st Hidden layer: 	2 node
> - 2nd Hidden layer: 	4 nodes
> - Output layer: 		5 nodes 

> **Warning**
> This function should be called once at the start of your project, repetitive calling will rewrite the weights.csv and bias.csv file thereby losing the previously computed weights and bias. It’s a better practice to keep this function in a separate file in your project, which will be executed only once till the end of the project.

**e.g.**

    import gargi_network as net
    # build the neural architecture
    net.build_network( [3, 2, 4, 3] )

## get_parameters( ) | return type: (dictionary, dictionary)

Retrieves the weight matrix and the bias vector from weights.csv and bias.csv respectively and returns it in the form of two dictionaries.
For weight dictionary: keys are the interface of the network and value are the weight matrix
For bias dictionary: keys are the layer of the network and values are the bias vector.

    import gargi_network as net
    # build the neural architecture
    net.build_network( [2, 1, 3] )
    # retrieve/cache the weight and bias 
    weight_dictionary, bias_vector = get_parameters()

    print(weight_dictionary)
    print(bias_vector)


Sample Output (the same result may not be expected ):

**weight_dictionary**:

    {0: [[-1.833882552398473], [-0.744739154869858]], 
     1: [[0.5060030345706763, -0.06417217225932671, -0.8651431114487439]]}

- **key** = 0 , **interface**: Input layer and the 1st hidden layer value = weight matrix of the dimension  2 x 1 
- **key** = 1, **interface**: 1st hidden layer and the output layer value = weight matrix of the dimension  1 x 3

**bias_vector**:

    {0: [0.0, 0.0], 
     1: [1.7049014856613136], 
     2: [0.14110336951682068, 0.08309189564643964, -0.03160026158283054]}

- **key** = 0: Layer 1 (Input layer), **value** = bias vector of dimension 2 x 1 (2 nodes)
- **key** = 1: Layer 2 (1st Hidden layer), **value** = bias vector of dimension 1 x 1 (1 node)
- **key** = 2: Layer 3 (Output layer), **value** = bias vector of dimension 3 x 1 (3 nodes)


## forward_propagation(*weight*, *bias*, *input*, *activation_func*) | return type: Numpy array

The input array/list is passed to the forward_propagation() function along with the weight and bias retrieved using get_parameters() function. An optional parameter for activation function is provided, by default the activation function is set to **SIGMOID** activation function, it can also be set to **RELU** and **STEP** activation function.

    import gargi_network as nn
    weight, bias = nn.get_parameters()
    op = nn.forward_propagation(w, b, [1, 2])
    print(op)
For input:
    
    [1, 2]
Output:
    
    [0.44384968 0.27767789]
> **Note**
> The output will not necessarily match with any of your run, but the dimension should match the output dimension provided at the build_network() function

To set the activation function as **SIGMOID**

    op = nn.forward_propagation(w, b, [1, 2])
or 

    op = nn.forward_propagation(w, b, [1, 2], 'sigmoid')
To set the activation function as **RELU**

    op = nn.forward_propagation(w, b, [1, 2], 'relu')

To set the activation function as **STEP**

    op = nn.forward_propagation(w, b, [1, 2], 'step')
## back_propagation(*input_matrix*, *target_matrix*, *weight*, *bias*, *activation_function*, *learning_rate*) | return type: tuple(List[List[]], List[List[]])
The input matrix along with the target matrix, weight, bias,  activation function and learning rate is passed to the back_propagation() function. The activation function and the learning rate are optional parameters that have the default value of **sigmoid** and **0.001** respectively.
