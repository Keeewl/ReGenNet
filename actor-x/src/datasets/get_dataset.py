def get_dataset(name="chi3d"):
    if name in ['chi3d', 'interx']:
        from .feeder_2p import Feeder_2P
        return Feeder_2P

def get_datasets(parameters):
    name = parameters["dataset"]
    parameters.update({'dataname': name})

    DATA = get_dataset(name)
    dataset = DATA(split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets
