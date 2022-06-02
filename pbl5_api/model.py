from torch import nn
import torch

from collections import namedtuple
import math

# from numba import cuda

# torch.cuda.empty_cache()

# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU6(inplace = True)
            
        self.downsample = downsample
        
    def forward(self, x):
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
                
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out += residual
        out = self.relu(out)
    
        return out


class ResNet(nn.Module):
  def __init__(self, config, output_dim):
      super(ResNet,self).__init__()
      block, n_blocks, channels = config
      self.in_channels = channels[0]

      assert len(n_blocks) == len(channels) == 4

      self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, 
                             kernel_size=7, stride=2, padding=3, bias=False)
      self.bn1 = nn.BatchNorm2d(num_features=self.in_channels, eps=1e-5, 
                                momentum=0.1)
      self.relu = nn.ReLU6(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
      self.layer1 = self._make_layer(block=block, n_blocks=n_blocks[0],
                                     out_channels=channels[0], stride=1)
      
      self.layer2 = self._make_layer(block=block, n_blocks=n_blocks[1],
                                     out_channels=channels[1], stride=2)
      
      self.layer3 = self._make_layer(block=block, n_blocks=n_blocks[2],
                                     out_channels=channels[2], stride=2)
      
      self.layer4 = self._make_layer(block=block, n_blocks=n_blocks[3],
                                     out_channels=channels[3], stride=2)
      
      self.avgpool = nn.AdaptiveAvgPool2d((1,1))

      self.fc = nn.Linear(512 * 4, output_dim)

      for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

      
  def _make_layer(self, block, n_blocks, out_channels, stride):
      downsample = None
      layers = []
      if stride != 1 or self.in_channels != out_channels * 4:
          conv = nn.Conv2d(self.in_channels, 4 * out_channels, kernel_size = 1, 
                          stride = stride, bias = False)
          bn = nn.BatchNorm2d(4 * out_channels)
          downsample = nn.Sequential(conv, bn)

      layers.append(block(self.in_channels, out_channels, stride, downsample))
      
      self.in_channels = 4 * out_channels
      for i in range(n_blocks - 1):
          layers.append(block(self.in_channels, out_channels))

      return nn.Sequential(*layers)

  def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      
      return x
