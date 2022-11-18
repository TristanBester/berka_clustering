import torch.nn.functional as F


def _multi_rec_loss(encoder_outputs, decoder_outputs):
    total_loss = 0

    for enc, dec in zip(encoder_outputs, decoder_outputs):
        total_loss += F.mse_loss(enc, dec)
    return total_loss


def multi_rec_loss_handler(encoder, embedding, decoder, x, device):
    encoder_outputs, h = encoder.layerwise_forward(x)
    z, _, _ = embedding(h, device)
    decoder_outputs = decoder.layerwise_forward(z)
    return _multi_rec_loss(encoder_outputs, decoder_outputs)
