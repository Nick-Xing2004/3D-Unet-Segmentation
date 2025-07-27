import torch.nn as nn
import torch.nn.functional as F
import torch

#the second version of the UNet3D model, with stronger receptive field. (6 encoding layers, 5 encoding layers for the first version)       [07/01/2025]

class UNet3D_2(nn.Module):
    # model intialization
    def __init__(self, in_channels=1, out_channels=6, init_features=32):
        super(UNet3D_2, self).__init__()
        features = init_features
        
        #encoder part
        #first layer： #in_channels=1 -> out_channels=32
        self.encoder1 = self._block(in_channels, features)  #conv-3d * 2
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  #down-sampling using 2*2*2 kernel, D/2, H/2, W/2
        
        #second layer #in_channels=32 -> out_channels=64
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        #third layer  #in_channels=64 -> out_channels=128
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        #fourth layer #in_channels=128 -> out_channels=256
        self.encoder4 = self._block(features * 4, features * 8)  #number of out-channels 256   2nd-dim=256
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        #fifth layer  #in_channels=256 -> out_channels=512
        self.encoder5 = self._block(features * 8, features * 16) 
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        #bottleneck part
        self.bottleneck = self._block(features * 16, features * 32)   #number of out-channels 1024 (largest receptive field)

        #decoder part
        self.upconv5 = nn.ConvTranspose3d(features * 32, features * 16, kernel_size=2, stride=2)  
        self.decoder5 = self._block(features * 32, features * 16) 

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8)    #further convolution (features infusion), prepare for the next upsampling

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
    

    # forward propagation 
    def forward(self, x):
        #encoding and down-sampling
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        #bottleneck formation
        bottleneck = self.bottleneck(self.pool5(enc5))      #features * 32

        #decoding and up-sampling
        dec5 = self.upconv5(bottleneck)   #features * 32 -> features * 16
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.decoder5(dec5)        #features * 16

        dec4 = self.upconv4(dec5)   #features * 16 -> features * 8
        dec4 = torch.cat([dec4, enc4], dim=1)     
        dec4 = self.decoder4(dec4)        #features * 8

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)        #features * 4

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)        #features * 2
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)        #features
        
        #final convolution to get the output
        output = self.final_conv(dec1)
        return output 
    

    # helper function for doing the convolution
    # operations within the function:
    # 1. doing convolution while doubles the number of features(out-channels)
    # 2. doing batch normalization over it
    # 3. applying ReLU activation function
    # 4. repeate the process one more time
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    # helper function for cropping the output of the encoder, to fit and unify the sizes of encoder output & decoder output 
    # for skip connection
    # def center_crop(self, encoder_output, decoder_output):
    #     _, _, D, H, W = decoder_output.shape
    #     d, h, w = encoder_output.shape[2:]
    #     return encoder_output[:, :, ]


#Function to create the model
def initialize_Unet3D_2(device, in_channels=1, out_channels=6, init_features=32):
    model = UNet3D_2(in_channels, out_channels, init_features)
    print(f'now using the second 2️⃣ version of the model!')
    return model #used to move the model to the specified device