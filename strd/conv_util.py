from torch.nn import Conv2d


def list_convs(model):
    res = []
    for name, mod in model.named_modules():
        if isinstance(mod, Conv2d):
            res.append(name)
    return res
