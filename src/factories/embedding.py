from src.modules.embedding import IdentityEmbedding, VAEEmbedding, VectorEmbedding


class IncompatabilityException(Exception):
    """Component incompatibility exception."""

    def __init__(self, message):
        super().__init__(message)


def embedding_factory(ae_name, pretext_loss, encoder_output_dim, embedding_dim):
    if pretext_loss == "mse":
        if ae_name == "lstm" or ae_name == "dtc":
            return IdentityEmbedding()
        else:
            return VectorEmbedding(
                in_features=encoder_output_dim, embedding_dim=embedding_dim
            )
    if pretext_loss == "multi_rec":
        if ae_name == "lstm" or ae_name == "dtc":
            raise IncompatabilityException(
                "RNN models are not compatible with DEPICT loss."
            )
        return VectorEmbedding(
            in_features=encoder_output_dim, embedding_dim=embedding_dim
        )
    if pretext_loss == "vae":
        if ae_name == "dtc":
            raise IncompatabilityException(
                "DTC encoder is not compatible with VAE loss."
            )
        return VAEEmbedding(in_features=encoder_output_dim, embedding_dim=embedding_dim)
