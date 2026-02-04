import torch


def configvalues_totensorlist(module, propertynames, device=None):
    """
    Docstring for configvalues_totensorlist


    :param module: Description
    :param propertynames: Description
    :param device: Description
    """

    if device is None:
        # get it from the module's weight
        if hasattr(module, "weight"):
            device = module.weight.device
        else:
            device = torch.device("cpu")
            # worst case scenario it will be
            # an error during forward pass
            # if there is no compatible device

    if device is not None:
        # assert whether the device is the same as module's weight
        # check if module has weight
        if hasattr(module, "weight"):
            if device != module.weight.device:
                print(
                    "Warning: the specified device is different from module weight device"
                )
                exit()

    values = []
    for attr in propertynames:
        v = getattr(module, attr)
        if isinstance(v, bool):
            v = torch.tensor([v], dtype=torch.bool, device=device)
        elif isinstance(v, int):
            v = torch.tensor([v], dtype=torch.int32, device=device)
        elif isinstance(v, bool):
            v = torch.tensor([v], dtype=torch.int32, device=device)
        elif isinstance(v, tuple):
            # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple

            v = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            print("v is neither int nor tuple. unexpected")
            exit()
        values.append(v)

    return values


def tensorlist_todict(values, propertynames):
    paramsdict = {}
    for i, n in enumerate(propertynames):
        v = values[i]
        if v.numel == 1:
            paramsdict[n] = v.item()
        else:
            alist = v.tolist()

            if len(alist) == 1:
                paramsdict[n] = alist[0]
            else:
                paramsdict[n] = tuple(alist)
    return paramsdict
