import torch.optim as optim


def pretrain_autoencoder(model, loader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for i in range(10):
        for x, _ in loader:
            loss = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
