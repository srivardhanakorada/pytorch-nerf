# from torchvision.models import resnet34

# import torch

# import torch.nn as nn

# import torch.nn.functional as F

# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet = resnet34(True)
#     def forward(self, x):
#         # pritn resnet34 structure
#         print(self.resnet)


# # Create an instance of the ImageEncoder class
# img_enc = ImageEncoder()
# img_enc(torch.randn(1, 3, 224, 224))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50


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
        print(feats1.shape, feats2.shape, feats3.shape, feats4.shape)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        print(latent_sz)
        # for i in range(len(latents)):
        #     latents[i] = F.interpolate(
        #         latents[i], latent_sz, mode="bilinear", align_corners=True
        #     )

        # latents = torch.cat(latents, dim=1)
        # print(latents.shape)
        # return latents
        return None

img_enc = ImageEncoder()
img_enc(torch.randn(1, 3, 128, 128))