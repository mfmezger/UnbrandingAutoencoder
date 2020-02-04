from torch  import nn

# modified version from: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=3),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
