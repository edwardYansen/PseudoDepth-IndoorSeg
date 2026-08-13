import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, UperNetForSemanticSegmentation, UperNetConfig
from collections import OrderedDict
import segmentation_models_pytorch as smp
from torchvision import models

# model = UNetRGBD(in_channels=4, num_classes=NUM_CLASSES)
class UNetRGBD(nn.Module):
    """
    U-Net model for RGB-D input with a pre-trained ResNet-50 encoder.
    """
    def __init__(self,
                in_channels=4,
                num_classes=14,
                partial_pretrained=True
                ):
        super(UNetRGBD, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.encoder0 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc0conv1",
                        nn.Conv2d(
                            in_channels,
                            num_classes,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "enc0norm1",
                        nn.BatchNorm2d(
                            num_classes,
                            eps=1e-05,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                    ("enc0relu1", nn.ReLU(inplace=True)),
                    (
                        "enc0conv2",
                        nn.Conv2d(
                            num_classes,
                            in_channels,
                            kernel_size=7,
                            stride=2,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (
                        "enc0norm2",
                        nn.BatchNorm2d(
                            in_channels,
                            eps=1e-05,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                    ("enc0relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.unet = smp.UnetPlusPlus(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
            activation="softmax"
        )

    def encoderTest(self, x: torch.Tensor):
        return self.encoder0(x)

    def forward(self, x: torch.Tensor):
        # encoder = self.encoder0(x)
        return self.unet(x)
    
    def setup_finetune_decoders(self):
        for name, p in self.unet.named_parameters():
            if (
                name.startswith("decoder")
                or name.startswith("upconv")
                or name.startswith("conv")
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def setup_finetune_decoders_bottleneck(self):
        self.setup_finetune_decoders()
        for name, p in self.unet.named_parameters():
            if name.startswith("bottleneck"):
                p.requires_grad = True
            else:
                p.requires_grad = False

# model = SegFormerDepth(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
class SegFormerDepth(nn.Module):
    """
    SegFormer adapted for RGB-D semantic segmentation.
    - Accepts 4-channel input (RGB + Depth)
    - Sets fixed image_size in the model config during initialization
    """

    def __init__(self, in_channels=4, num_classes=19,
                 image_size=(512, 512),
                 backbone="nvidia/segformer-b3-finetuned-ade-512-512"):
        super(SegFormerDepth, self).__init__()

        # Load pretrained config and override image_size + num_labels
        config = SegformerConfig.from_pretrained(backbone)
        config.num_labels = num_classes
        self.image_size = image_size  # fixed resolution set here

        # Build model with updated config
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            config=config,
            ignore_mismatched_sizes=True
        )

        if in_channels == 4:
            # Modify first patch embedding to accept 4 channels
            stem = self.model.segformer.stages[0].patch_embeddings.proj
            old_weight = stem.weight.data
            out_channels, _, kH, kW = old_weight.shape

            new_stem = nn.Conv2d(4, out_channels, kernel_size=7,
                                stride=4, padding=3, bias=False)

            with torch.no_grad():
                # Copy RGB weights
                new_stem.weight[:, :3, :, :] = old_weight
                # Initialize depth channel as mean of RGB filters
                depth_init = old_weight.mean(dim=1, keepdim=True)
                new_stem.weight[:, 3:4, :, :] = depth_init

            self.model.segformer.stages[0].patch_embeddings.proj = new_stem

    def forward(self, pixel_values):
        """
        Forward pass
        - pixel_values: (B, 4, H, W) RGB-D tensor
        Returns:
        - logits: (B, num_classes, H_out, W_out) with resolution set by config.image_size
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=self.image_size,
                            mode="bilinear", align_corners=False)

        return logits

def RGBD_resizeBackbone(in_channels: int, model: UperNetForSemanticSegmentation) -> nn.Conv2d:    
    pe = model.backbone.swin.embeddings.patch_embeddings
    assert hasattr(pe, "projection"), "SwinPatchEmbeddings should expose `.proj` Conv2d"
    old_conv = pe.projection  # nn.Conv2d(in_channels=3, kernel_size=4, stride=4, ...)

    new_conv = nn.Conv2d(
        in_channels=in_channels,                                   # <-- RGB + Depth
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Initialize new conv: copy RGB weights; initialize depth channel reasonably
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        # Use mean of RGB filters for depth channel (simple & stable init)
        new_conv.weight[:, 3:4] = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
            
    return new_conv

# modelRaw = SwinTransformerPretrained(in_channels=4, num_classes=NUM_CLASSES)
class SwinTransformerPretrained(nn.Module):
    """
    Swin-Transformer model for RGB-D input with a pre-trained
    """
    def __init__(self,
                in_channels=4,
                num_classes=14,
                partial_pretrained=True
                ):
        super(SwinTransformerPretrained, self).__init__()
        # 1) Start from the OpenMMLab UPerNet+Swin config and override num_labels
        cfg = UperNetConfig.from_pretrained(
            "openmmlab/upernet-swin-large",
            num_labels=num_classes,                       # <-- target number of classes
            id2label={i: str(i) for i in range(13)},  # optional: map ids to names later
            label2id={str(i): i for i in range(13)},
        )

        # 2) Load pretrained weights, but allow final head size to change
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-swin-large",
            config=cfg,
            ignore_mismatched_sizes=True   # <-- re-initializes classifier to 13 channels
        )
        if in_channels == 4:
            # Replace
            self.model.backbone.swin.embeddings.patch_embeddings.projection = RGBD_resizeBackbone(in_channels, self.model)

    def forward(self, x: torch.Tensor):
        # encoder = self.encoder0(x)
        return self.model(x).logits
    


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-sized inputs
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# model = UNetResNet34_RGBD(num_classes=NUM_CLASSES)
class UNetResNet34_RGBD(nn.Module):
    """
    U-Net with a ResNet34 encoder adapted for 4-channel RGBD input.
    The first conv is inflated from 3 -> 4 channels by copying ImageNet weights
    and initializing the 4th channel as the mean of RGB kernels.
    """
    def __init__(self, in_channels=4, num_classes=1, pretrained=True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        if in_channels == 4:
            # Replace
            # Inflate stem conv1 from 3->4 channels while keeping pretrained weights
            old_conv1 = resnet.conv1  # (64, 3, 7, 7)
            new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if pretrained:
                    new_conv1.weight[:, :3] = old_conv1.weight  # copy RGB
                    # initialize D channel as mean of RGB filters
                    new_conv1.weight[:, 3:4] = old_conv1.weight.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
            resnet.conv1 = new_conv1

        # Encoder
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2, C64
        self.pool0 = resnet.maxpool                                       # /4
        self.enc1 = resnet.layer1  # /4,  C64
        self.enc2 = resnet.layer2  # /8,  C128
        self.enc3 = resnet.layer3  # /16, C256
        self.enc4 = resnet.layer4  # /32, C512

        # Decoder
        self.up4 = UpBlock(512, 256, 256)  # /32 -> /16
        self.up3 = UpBlock(256, 128, 128)  # /16 -> /8
        self.up2 = UpBlock(128, 64, 64)    # /8  -> /4
        self.up1 = UpBlock(64, 64, 64)     # /4  -> /2

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # /2 -> /1
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (N, 4, H, W) RGBD
        x0 = self.enc0(x)       # /2
        x0p = self.pool0(x0)    # /4
        x1 = self.enc1(x0p)     # /4
        x2 = self.enc2(x1)      # /8
        x3 = self.enc3(x2)      # /16
        x4 = self.enc4(x3)      # /32

        d4 = self.up4(x4, x3)   # /16
        d3 = self.up3(d4, x2)   # /8
        d2 = self.up2(d3, x1)   # /4
        d1 = self.up1(d2, x0)   # /2
        d0 = self.up0(d1)       # /1

        return self.out_conv(d0)
