from src.modules.pretext import (
    mse_loss_handler,
    multi_rec_loss_handler,
    vae_loss_handler,
)


def pretext_factory(pretext):
    if pretext == "mse":
        return mse_loss_handler
    if pretext == "multi_rec":
        return multi_rec_loss_handler
    if pretext == "vae":
        return vae_loss_handler
