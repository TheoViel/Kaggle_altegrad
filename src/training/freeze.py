def freeze(model):
    """
    Freezes a model
    
    Arguments:
        model {torch model} -- Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(modem):
    """
    Unfreezes a model
    
    Arguments:
        modem {torch model} -- Model to unfreeze
    """
    for param in modem.parameters():
        param.requires_grad = True


def freeze_layer(model, name):
    """
    Freezes layers in a model
    Arguments:
        model {torch model} -- Model to freeze layers in
        name {string} -- String contained by the name of layer to freeze
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model, name):
    """
    Unfreezes layers in a model
    Arguments:
        model {torch model} -- Model to unfreeze layers in
        name {string} -- String contained by the name of layer to unfreeze
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True