from src.modules.decoders import DTCDecoder, FCNNDecoder, LSTMDecoder, ResNetDecoder


def decoder_factory(ae_name, seq_len, embedding_dim):
    if ae_name == "fcnn":
        return FCNNDecoder(seq_len=seq_len, embedding_dim=embedding_dim)
    if ae_name == "resnet":
        return ResNetDecoder(seq_len=seq_len, embedding_dim=embedding_dim)
    if ae_name == "lstm":
        return LSTMDecoder(seq_len=seq_len, embedding_dim=embedding_dim)
    if ae_name == "dtc":
        return DTCDecoder(output_size=seq_len, deconv_kernel=10, deconv_stride=5)
