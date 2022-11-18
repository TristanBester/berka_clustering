import torch.optim as optim


def pretrain_autoencoder(model, loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_count = 0 

    model = model.to(device)

    while batch_count < 1000:
        for x, _ in loader:
            x = x.to(device)
            loss = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1

            if batch_count > 1000:
                break
            # print(batch_count, loss.item())

    