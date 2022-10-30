import torch.optim as optim

from ..models import Autoencoder


def pretrain_autoencoder(config, loader):
    model = Autoencoder(config)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for i in range(10):
        for x, _ in loader:
            loss = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
    return model
