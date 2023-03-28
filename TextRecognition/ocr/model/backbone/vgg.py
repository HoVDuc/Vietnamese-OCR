import torch
import torch.nn as nn 
import torchvision


class VGG(nn.Module):
    
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5) -> None:
        super(VGG, self).__init__()
        
        assert name in ['vgg11_bn', 'vgg1_bn'], "Name model not supported"
        
        if name == 'vgg11_bn':
            cnn = torchvision.models.vgg11_bn(pretrained=pretrained)
        else:
            cnn = torchvision.models.vgg19_bn(pretrained=pretrained)
            
        pool_idx = 0
        
        for i, m in enumerate(cnn.features):
            if isinstance(m, nn.MaxPool2d):
                cnn.features[i] = nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
        
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv1x1 = nn.Conv2d(512, hidden, 1)
    
    def forward(self, x):
        out = self.features(x)
        out = self.dropout(out)
        out = self.last_conv1x1(out)
        
        out = out.transpose(-1, -2)
        out = out.flatten(2)
        out = out.premute(-1, 0, 1)
        
        return out
    
def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return VGG('vgg11_bn', ss, ks, hidden, pretrained, dropout)

def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return VGG('vgg19_bn', ss, ks, hidden, pretrained, dropout)