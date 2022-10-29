import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        embedding_dim,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel,
            stride=cnn_stride,
        )
        self.max_pool = nn.MaxPool1d(
            kernel_size=mp_kernel,
            stride=mp_stride,
        )

        self.lstm_layer_one = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=embedding_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_two = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=1,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x = F.leaky_relu(self.cnn(x))
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm_layer_one(x)
        x = x[:, :, : self.embedding_dim] + x[:, :, : self.embedding_dim]

        x, (_, _) = self.lstm_layer_two(x)
        x = (x[:, :, 0] + x[:, :, 1]).unsqueeze(-1)
        x = x.permute(0, 2, 1)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, deconv_kernel, deconv_stride) -> None:
        super().__init__()
        self.output_size = output_size
        self.kernel = deconv_kernel
        self.stride = deconv_stride

        self._calcuate_required_kernel()

        upsample = int((self.output_size + self.stride - self.kernel) / self.stride)

        self.upsample = nn.Upsample(size=(upsample,))
        self.deconv = nn.ConvTranspose1d(
            1, 1, kernel_size=self.kernel, stride=self.stride
        )

    def _calcuate_required_kernel(self):
        while True:
            l_up = (self.output_size + self.stride - self.kernel) / self.stride

            if l_up % 1 == 0:
                break
            else:
                self.kernel -= 1

    def forward(self, x):
        x = self.upsample(x)
        x = F.leaky_relu(self.deconv(x))
        return x


class DTCAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        cnn_channels=50,
        cnn_kernel=5,
        cnn_stride=1,
        mp_kernel=10,
        mp_stride=10,
        embedding_dim=10,
        deconv_kernel=10,
        deconv_stride=5,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            cnn_stride=cnn_stride,
            mp_kernel=mp_kernel,
            mp_stride=mp_stride,
            embedding_dim=embedding_dim,
        )
        self.decoder = Decoder(
            output_size=seq_len,
            deconv_kernel=deconv_kernel,
            deconv_stride=deconv_stride,
        )

    def forward(self, x):
        l = self.encoder(x)
        return self.decoder(l)
