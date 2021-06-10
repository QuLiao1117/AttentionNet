import torch
import torch.nn as nn
from vit_pytorch import ViT

class Transformer(nn.Module):
    def __init__(self,image_size=32,patch_size=4,num_classes=10):
        super(Transformer, self).__init__()
        # self.preconv=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 256, 3, stride=1, padding=1)
        )

        self.vit=ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            channels=256,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        # pre=self.preconv(x)
        pre = self.conv_embedding(x)
        out = self.vit(pre)
        return out


def test():
    net = Transformer()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
