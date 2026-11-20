def calculate_parameters(layers: list[dict]) -> int:
    parameter_num =0 
    for layer in layers:
        if(layer['type'] == 'dense'):
            parameter_num += layer['input_size'] * layer['output_size']
            parameter_num += layer['output_size']
            if 'bias' in layer:
                if(layer['bias'] == False):
                    
                    parameter_num -= layer['output_size']
        else:
            parameter_num += layer['kernel_size']*layer['kernel_size']*layer['in_channels']*layer['out_channels']
            if(layer['bias'] == True):
                parameter_num += layer['out_channels']
    return parameter_num

            