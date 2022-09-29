"""
Function to load weight to spesific models
"""
def backbone_load(model,model_weight,keys=['input_conv','unet','offset_linear', 'output_layer', 'semantic_linear']):
    processed_dict = {}
    for k in net_weighth.keys(): 
        decomposed_key = k.split(".")[0]
        if(decomposed_key in keys):
            processed_dict[k] = net_weighth[k] 
    model_dict = model.state_dict()
    model_dict.update(model_weight)
    model.load_state_dict(model_dict)
    return model