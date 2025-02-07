from torch.nn import Module, Sequential, Linear, \
    ReLU, PReLU, Tanh

def make_mlp(input_size: int, output_size: int, hidden_size: int = None,
             num_layers: int = 1, bias: bool = True, activation: str = None,
             device: str ='cpu') -> Sequential:
    if num_layers == 1:
        return _linear_w_activation(in_features=input_size, out_features=output_size,
                                           bias=bias, activation=activation, device=device)

    hidden_size = hidden_size if hidden_size is not None else (input_size * 2)
    layers = []
    # Input Layer
    layers.append(_linear_w_activation(in_features=input_size, out_features=hidden_size,
                                           bias=bias, activation=activation, device=device))

    # Hidden Layers
    for _ in range(num_layers-2):
        layers.append(_linear_w_activation(in_features=hidden_size, out_features=hidden_size,
                                           bias=bias, activation=activation, device=device))

    # Output Layer
    layers.append(_linear_w_activation(in_features=hidden_size, out_features=output_size,
                                           bias=bias, activation=activation, device=device))
    return Sequential(*layers)

def _linear_w_activation(in_features: int, out_features: int, bias: bool = True,
                         activation: str = None, device: str = 'cpu') -> Sequential:
    layers = []
    layers.append(Linear(in_features=in_features, out_features=out_features, bias=bias, device=device))
    if activation is not None:
        layers.append(get_activation_func(activation, device=device))
    return Sequential(*layers)

def get_activation_func(name, device: str = 'cpu') -> Module:
    if name == 'relu':
        return ReLU()
    if name == 'prelu':
        return PReLU(device=device)
    if name == 'tanh':
        return Tanh()
    return None
