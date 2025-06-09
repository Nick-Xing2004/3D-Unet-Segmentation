import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        
        #encoder
        self.encoder1 = self._block(in_channels, features)  #conv-3d * 2
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  #down-sampling using 2*2*2 kernel, D/2, H/2, W/2

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)  #number of out-channels 256
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        #bottleneck part
        self.bottleneck = self._block(features * 8, features * 16)

        #decoder
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8)    #further convolution (features infusion), prepare for the next upsampling

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
    
    def forward(self, x):








