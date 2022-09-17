# Code adapted from: https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

import torch
import torch.nn as nn
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        layer_output = self.conv(inputs)
        pool_output = self.pool(layer_output)

        return layer_output, pool_output


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip_inputs):
        x = self.upsample(inputs)
        x = torch.cat([x, skip_inputs], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.encoder_1 = encoder_block(input_channels, 64)
        self.encoder_2 = encoder_block(64, 128)
        self.encoder_3 = encoder_block(128, 256)
        self.encoder_4 = encoder_block(256, 512)

        self.bottleneck = conv_block(512, 1024)

        self.decoder_1 = decoder_block(1024, 512)
        self.decoder_2 = decoder_block(512, 256)
        self.decoder_3 = decoder_block(256, 128)
        self.decoder_4 = decoder_block(128, 64)

        self.classifier = nn.Conv2d(64, output_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        layer_output1, pool_output1 = self.encoder_1(inputs)
        layer_output2, pool_output2 = self.encoder_2(pool_output1)
        layer_output3, pool_output3 = self.encoder_3(pool_output2)
        layer_output4, pool_output4 = self.encoder_4(pool_output3)

        bottleneck_output = self.bottleneck(pool_output4)

        decoder_output1 = self.decoder_1(bottleneck_output, layer_output4)
        decoder_output2 = self.decoder_2(decoder_output1, layer_output3)
        decoder_output3 = self.decoder_3(decoder_output2, layer_output2)
        decoder_output4 = self.decoder_4(decoder_output3, layer_output1)

        output = self.classifier(decoder_output4)
        
        output = self.sigmoid(output)

        return output
