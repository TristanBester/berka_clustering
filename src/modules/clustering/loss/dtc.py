import torch
import torch.nn.functional as F


def dtc_loss_handler(clustering_layer, decoder, x):
    (z, _, _), Q, P = clustering_layer(x)
    x_prime = decoder(z)
    log_Q = torch.log(Q)
    log_P = torch.log(P)

    kl_div_loss = F.kl_div(log_Q, log_P, reduction="batchmean", log_target=True)
    mse_loss = F.mse_loss(x_prime, x)
    return mse_loss + kl_div_loss
