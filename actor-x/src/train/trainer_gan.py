import torch
from tqdm import tqdm


def train_or_test(model, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in model.losses}

    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            batch = {key: val.to(device) for key, val in batch.items()}

            # forward pass
            iter_info = model(batch)
            # lossD = iter_info['lossD']
            # lossG = iter_info['lossG']

    return iter_info


def train(model, iterator, device):
    return train_or_test(model, iterator, device, mode="train")


def test(model, iterator, device):
    return train_or_test(model, iterator, device, mode="test")
