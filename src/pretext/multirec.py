import torch.nn.functional as F


def multi_rec_loss(encoder_outputs, decoder_outputs):
    total_loss = 0

    for enc, dec in zip(encoder_outputs, decoder_outputs):
        total_loss += F.mse_loss(enc, dec)
    return total_loss
