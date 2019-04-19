
import os
import sys
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import autograd
from torchvision import models
from torch.autograd import Variable


############################# Open-Set Networks
class classifierM(nn.Module):
	def __init__(self, latent_size=100, no_closed=1):
		super(classifierM, self).__init__()
		self.fc1 = nn.Linear( latent_size, no_closed)

	def forward(self, input):
		out = self.fc1(input)
		return out

class encoderM(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(128)

        self.bn4 = nn.InstanceNorm2d(128)
        self.bn5 = nn.InstanceNorm2d(128)
        self.bn6 = nn.InstanceNorm2d(128)

        self.bn7 = nn.InstanceNorm2d(128)
        self.bn8 = nn.InstanceNorm2d(128)
        self.bn9 = nn.InstanceNorm2d(128)
        self.bn10 = nn.InstanceNorm2d(128)

        self.fc1 = nn.Linear(128*2*2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)

    def forward(self, x, output_scale=1):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        if output_scale == 8:
            x = self.conv_out_6(x)
            x = x.view(batch_size, -1)
            return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        if output_scale == 4:
            x = self.conv_out_9(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, 4*4)
            return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        if output_scale == 2:
            x = self.conv_out_10(x)
            x = x.view(batch_size, -1)
            return x

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        
        return x

class generatorM(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, n=6, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.n = n
        self.mul = nn.Linear(n, 1)
        self.add  = nn.Linear(n, 16*self.latent_size)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1)

        self.bn1 = nn.InstanceNorm2d(512)
        self.bn2 = nn.InstanceNorm2d(512)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)

    def forward(self, x, input_scale=1):
        
        if input_scale <= 1:

            x = self.fc1(x)
            x = x.resize(self.batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(self.batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:

            x = x.view(self.batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(self.batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        
        # 128 x 16 x 16
        x = self.conv5(x)
        
        # 3 x 32 x 32
        x = nn.Tanh()(x)
        return x

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if m.bias is not None:
			m.bias.data.fill_(0.0)

############################# OOD Networks
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

## implementation of densenet cifar taken from bamos@github
class BottleneckDense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(BottleneckDense, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayerDense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayerDense, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class TransitionDense(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(TransitionDense, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet10(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet10, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = TransitionDense(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = TransitionDense(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(BottleneckDense(nChannels, growthRate))
            else:
                layers.append(SingleLayerDense(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return out

class DenseClassifier10(nn.Module):
	def __init__(self, latent_size=342, no_closed=10):
		super(DenseClassifier10, self).__init__()
		self.fc1 = nn.Linear(latent_size, no_closed)

	def forward(self, input):
		# xf  = input.view(-1, self.ngf*16*2*2)
		out = self.fc1(input)
		return out

############################# Ablation Network
class DCCA_Encoder(nn.Module):
	def __init__(self):
		super(DCCA_Encoder, self).__init__()
		self.no_class = 15
		self.encoder = nn.Sequential(
			# nn.Dropout2d(0.5),
			nn.Conv2d(3, 32, 3, stride=2, padding=(2,2)),
			# nn.BatchNorm2d(32),
			nn.ReLU(True),
			# nn.Dropout2d(0.5),
			nn.Conv2d(32, 64, 3, stride=2, padding=(2,2)),
			# nn.BatchNorm2d(64),
			nn.ReLU(True),
			# nn.Dropout2d(0.5),
			nn.Conv2d(64, 128, 3, stride=2),
			# nn.BatchNorm2d(128),
			nn.ReLU(True)
		)
		self.fc = nn.Linear(8*8*128,self.no_class)

	def forward(self, x):
		out = x
		out = self.encoder(out)
		out = out.view(-1,8*8*128)
		out = self.fc(out)
		return out

class DCCA_Decoder(nn.Module):
	def __init__(self):
		super(DCCA_Decoder, self).__init__()
		self.no_class = 15
		self.decoder = nn.Sequential(
			# nn.Dropout2d(0.5),
			nn.ConvTranspose2d(128, 64, 3, stride=2),
			# nn.BatchNorm2d(64),
			nn.ReLU(True),
			# nn.Dropout2d(0.5),
			nn.ConvTranspose2d(64, 32, 3, stride=2, padding=(1,1),  output_padding=(0,0)),
			# nn.BatchNorm2d(32),
			nn.ReLU(True),
			# nn.Dropout2d(0.5),
			nn.ConvTranspose2d(32, 3, 3, stride=2, padding=(2,2), output_padding=(1,1)),
			nn.Tanh()
		)
		self.fc = nn.Linear(self.no_class,128*8*8)
		# self.bn = nn.BatchNorm1d(128*3*3)

	def forward(self, x):
		out = x
		out = self.fc(out)
		# out = self.bn(out)
		out = out.view(-1,128,8,8)
		out = self.decoder(out)
		return out

class DCCA_Label_Classifier(nn.Module):
	def __init__(self):
		super(DCCA_Label_Classifier, self).__init__()
		self.no_class = 15
		self.clf = nn.Sequential(
			# nn.ReLU(True),
			nn.Linear(self.no_class, self.no_class)
		)
	
	def forward(self, x):
		out = self.clf(x)
		return out









