from src.modules.encoders import DTCEncoder, FCNNEncoder, LSTMEncoder, ResNetEncoder


def encoder_factory(ae_name, seq_len, embedding_dim):
    if ae_name == "fcnn":
        return FCNNEncoder(seq_len=seq_len, embedding_dim=embedding_dim)
    if ae_name == "resnet":
        return ResNetEncoder(
            in_channels=1,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            pooling_kernel=10,
        )
    if ae_name == "lstm":
        return LSTMEncoder(embedding_dim=embedding_dim)
    if ae_name == "dtc":
        return DTCEncoder(
            input_dim=1,
            cnn_channels=50,
            cnn_kernel=5,
            cnn_stride=2,
            mp_kernel=10,
            mp_stride=5,
            embedding_dim=embedding_dim,
        )
