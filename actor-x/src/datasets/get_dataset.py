def get_dataset(name="ntu13"):
    if name == "ntu13":
        from .ntu13 import NTU13
        return NTU13
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == 'ntu120_2p_smpl' or name == 'ntu120_1p_smpl' or name == 'chi3d' or name == 'ntu120_2p_smplx' or name == 'hhi':
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
