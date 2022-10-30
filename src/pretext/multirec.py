import torch.nn.functional as F


def multi_rec_loss(encoder_outputs, decoder_outputs):
    total_loss = 0

    for enc, dec in zip(encoder_outputs, decoder_outputs):
        total_loss += F.mse_loss(enc, dec)
    return total_loss


def multi_rec_loss_handler(model, x):
    encoder_outputs, decoder_outputs = model.layerwise_forward(x)
    return multi_rec_loss(encoder_outputs, decoder_outputs)
