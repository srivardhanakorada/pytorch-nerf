import torch.nn as nn

from torchvision.models import inception_v3
import torch.nn.functional as F

# class ImageEncoderold(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet = resnet50(True)

#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         feats1 = self.resnet.relu(x)
#         feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
#         feats3 = self.resnet.layer2(feats2)
#         feats4 = self.resnet.layer3(feats3)
#         latents = [feats1, feats2, feats3, feats4]
#         latent_sz = latents[0].shape[-2:]
#         for i in range(0,len(latents)-1):
#             latents[i] = F.interpolate(
#                 latents[i], latent_sz, mode="bilinear", align_corners=True
#             )
#         latents = torch.cat(latents[0:3], dim=1)
#         return latents

# write using inception_v3
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True)

    def forward(self, x):
        x = self.inception.Conv2d_1a_3x3(x)
        feats1 = x
        x = self.inception.Conv2d_2a_3x3(x)
        feats2 = x
        x = self.inception.Conv2d_2b_3x3(x)
        feats3 = x
        x = self.inception.Conv2d_3b_1x1(x)
        feats4 = x
        # x = self.inception.Conv2d_4a_3x3(x)
        # x = self.inception.Mixed_5b(x)
        # x = self.inception.Mixed_5c(x)
        # x = self.inception.Mixed_5d(x)
        # feats2 = x
        latents = [feats1, feats2, feats3, feats4]
        latent_sz = 64
        for i in range(0,len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )
        latents = torch.cat(latents, dim=1)
        return latents # output shape is 1*368*64*64
    
# create random input of shape 1*3*128*128 and pass to above image encoder
# print the output shape
import torch
x = torch.randn(1, 3, 128, 128)
model = ImageEncoder()
output = model(x)