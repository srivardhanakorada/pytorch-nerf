import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34, resnet50


class ImageEncoder_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)
        print(feats1.shape, feats2.shape, feats3.shape, feats4.shape)
        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return latents
    
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)
        # print(feats1.shape, feats2.shape, feats3.shape, feats4.shape)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(1,len(latents)-1):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )
        t1 = torch.cat([latents[1], latents[1]], dim=1)
        t = (t1 + latents[2])//2
        return t
