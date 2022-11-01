import torch.optim as optim


def pretrain_autoencoder(model, loader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_count = 0 

    while batch_count < 1000:
        for x, _ in loader:
            loss = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
    