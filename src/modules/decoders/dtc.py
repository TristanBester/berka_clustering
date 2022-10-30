import torch.nn as nn
import torch.nn.functional as F


class DTCDecoder(nn.Module):
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
