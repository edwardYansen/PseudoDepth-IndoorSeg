# ------------------------------------------------------------------------------
# Copyright (c) 2026 BINUS. All rights reserved.
#
# This code contains modifications and segments adapted from the PDDM project:
# PDDM: Pseudo Depth Diffusion Model for RGB-PD Semantic Segmentation 
# Based in Complex Indoor Scenes (AAAI 2025).
# https://github.com/Oleki-xxh/PDDM
#
# Original PDDM Authors: Xinhua Xu, Hong Liu, Jianbing Wu, Jinfu Liu
# ------------------------------------------------------------------------------
# Upstream Attribution & Compliance Note:
# Parts of this implementation are heavily derived from NVIDIA's ODISE repo, 
# which is governed by the NVIDIA Source Code License for ODISE. 
# As a result, this derivative work must follow those terms:
# - Maintained "AS IS" without warranties.
# - Intended strictly for NON-COMMERCIAL research or evaluation purposes.
#
# Original ODISE Code: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES.
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modifications for PseudoDepthAggregationModule by Edward B.Y.
# ------------------------------------------------------------------------------


from .DepthConstSet import DepthConstSet
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import random
import cv2
from PIL import Image # Import PIL for Image.fromarray
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
from diffusers import MarigoldDepthPipeline

def setDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"

def convertNpToImg(img:np.ndarray):
    img = Image.fromarray(img)
    return img

def invertDepth(output_numpy):    
    # Normalize the numpy array to the range [0, 1]
    normalized_output = (output_numpy - output_numpy.min()) / (output_numpy.max() - output_numpy.min())

    # *** INVERT THE DEPTH VALUES HERE ***
    # This makes 0 closest and 1 farthest
    inverted_output = 1 - normalized_output

    return inverted_output

def Transform_data_depth(images):
    listDepth1 = []
    listDepth2 = []
    listDepth3 = []
    anythingV2, image_processor = depthAnythingModel()
    marigold = depthMarigoldModel()
    midas, midas_transform = depthMidasModel()

    for _, image in enumerate(images):
        img = cv2.rotate(image.transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE) # RGB image (H, W, C)        
        # (depthMarigold, depthMidas, depthAnything) = create_pseudo_depth(img)
        # 2. Convert the NumPy array to a PIL Image.
        # The Marigold pipeline expects a PIL Image object as input.
        # input_image = Image.fromarray(img.astype(np.uint8))
        input_image = convertNpToImg(img)
        
        inputs_anythingv2 = image_processor(images=input_image, return_tensors="pt").to(setDevice())
        inputs_midas = midas_transform(img).to(setDevice())

        with torch.no_grad():
            outputs_anythingV2 = anythingV2(**inputs_anythingv2)
            prediction_anythingV2 = outputs_anythingV2.predicted_depth
            prediction_midas = midas(inputs_midas)

        result_anythingv2 = torch.nn.functional.interpolate(
            prediction_anythingV2.unsqueeze(1),
            size=input_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        result_midas = torch.nn.functional.interpolate(
            prediction_midas.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        result_depthMarigold = marigold(
            input_image,
            # ensemble_size=10,
            # num_inference_steps=20,
            # show_progress_bar=True,
            batch_size=8,
            output_type = "np"
        )
        depthMarigold_map_numpy = result_depthMarigold.prediction.squeeze()
        depthAnything=invertDepth(result_anythingv2.squeeze().cpu().numpy())
        depthMidas=invertDepth(result_midas.squeeze().cpu().numpy())

        listDepth1.append(depthMarigold_map_numpy)
        listDepth2.append(depthMidas)
        listDepth3.append(depthAnything)
    
    return listDepth1, listDepth2, listDepth3

def Transform_single_data_depth(img):
    anythingV2, image_processor = depthAnythingModel()
    marigold = depthMarigoldModel()
    midas, midas_transform = depthMidasModel()

    # img = cv2.rotate(rgb.transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE) # RGB image (H, W, C)        
    # (depthMarigold, depthMidas, depthAnything) = create_pseudo_depth(img)
    # 2. Convert the NumPy array to a PIL Image.
    # The Marigold pipeline expects a PIL Image object as input.
    # input_image = Image.fromarray(img.astype(np.uint8))
    # input_image = convertNpToImg(img)
    input_image = img.permute(1, 2, 0)
    # array = img.numpy()
    img_midas = input_image.numpy()
    
    inputs_anythingv2 = image_processor(images=input_image, return_tensors="pt").to(setDevice())
    inputs_midas = midas_transform(img_midas).to(setDevice())

    with torch.no_grad():
        outputs_anythingV2 = anythingV2(**inputs_anythingv2)
        prediction_anythingV2 = outputs_anythingV2.predicted_depth
        prediction_midas = midas(inputs_midas)

    result_anythingv2 = torch.nn.functional.interpolate(
        prediction_anythingV2.unsqueeze(1),
        size=input_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    
    result_midas = torch.nn.functional.interpolate(
        prediction_midas.unsqueeze(1),
        size=img_midas.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    result_depthMarigold = marigold(
        img,
        # ensemble_size=10,
        # num_inference_steps=20,
        # show_progress_bar=True,
        batch_size=8,
        output_type = "np"
    )
    depthMarigold_map_numpy = result_depthMarigold.prediction.squeeze()
    depthAnything=invertDepth(result_anythingv2.squeeze().cpu().numpy())
    depthMidas=invertDepth(result_midas.squeeze().cpu().numpy())

    return depthMarigold_map_numpy, depthAnything, depthMidas

class PseudoDepthCreation():
    def __init__(self, transform_depth_dict = {}):
        self.transform_depth_dict = transform_depth_dict
        if transform_depth_dict is None or transform_depth_dict == {}:
            self.transform_depth_dict = {DepthConstSet().marigold, DepthConstSet().midas, DepthConstSet().depthAnythingV2}
        self.anythingV2, self.anythingV2_processor = depthAnythingModel()
        self.marigold = depthMarigoldModel()
        self.midas, self.midas_transform = depthMidasModel()
        self.device = setDevice()
    
    def AnythingV2_Depth(self, img):
        inputs_anythingv2 = self.anythingV2_processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs_anythingV2 = self.anythingV2(**inputs_anythingv2)
            prediction_anythingV2 = outputs_anythingV2.predicted_depth

        result_anythingv2 = torch.nn.functional.interpolate(
            prediction_anythingV2.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        depthAnything=invertDepth(result_anythingv2.squeeze().cpu().numpy())
        return depthAnything
        
    def Midas_Depth(self, img):
        inputs_midas = self.midas_transform(img).to(self.device)

        with torch.no_grad():
            prediction_midas = self.midas(inputs_midas)
            
        result_midas = torch.nn.functional.interpolate(
            prediction_midas.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depthMidas=invertDepth(result_midas.squeeze().cpu().numpy())
        return depthMidas

    def Marigold_Depth(self, img):
        result_depthMarigold = self.marigold(
            img,
            # ensemble_size=10,
            # num_inference_steps=20,
            # show_progress_bar=False,
            batch_size=4,
            output_type = "np"
        )
        
        depthMarigold_map_numpy = result_depthMarigold.prediction.squeeze()
        return depthMarigold_map_numpy
    
    def CreateListPseudoDepth(self, nyu_semantic13):
        MarigoldPseudoDepth=[]
        MidasPseudoDepth=[]
        AnythingV2PseudoDepth=[]

        for img, _ in nyu_semantic13:
            img = TransformImg(img)
            img_tp = img.transpose(1, 2, 0)
            
            if DepthConstSet().marigold in self.transform_depth_dict:
                MarigoldPseudoDepth.append(self.Marigold_Depth(img_tp))
            if DepthConstSet().midas in self.transform_depth_dict:
                MidasPseudoDepth.append(self.Midas_Depth(img_tp))
            if DepthConstSet().depthAnythingV2 in self.transform_depth_dict:
                AnythingV2PseudoDepth.append(self.AnythingV2_Depth(img_tp))
        
        return MarigoldPseudoDepth, MidasPseudoDepth, AnythingV2PseudoDepth


def TransformsDepth(depthInput, transform=None, depth_norm=10):
    if not transform: return depthInput
    
    depthNp = depthInput
    if(type(depthInput) is not np.ndarray):
        depthNp = depthInput.numpy()
    # For depth, scale to 0-255 for PIL conversion, then convert back after ToTensor
    depth_scaled_for_pil = np.clip(depthNp / depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_scaled_for_pil, mode='L') 

    # Reset random state and apply same transforms to depth and label
    depth = transform(depth_pil) # Apply transform to depth (PIL -> Tensor)
    depth = (depth / 255.0) * depth_norm # Scale depth back to original range
    
    # Ensure depth map has a channel dimension (1, H, W)
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    
    return depth

def depthAnythingModel():    
    # 2. Load the model and its processor
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    # Note: The model you initialized with a configuration above
    # does not have the pre-trained weights. You need to load them
    # using from_pretrained()
    model = DepthAnythingForDepthEstimation.from_pretrained(model_name)

    # Move the model to GPU if available
    device = setDevice()
    model.to(device)

    return model, image_processor

def depthMarigoldModel(): 
    # 3. Load the Marigold pipeline.
    device = setDevice()
    marigoldPipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        source="local",
        use_fast=True
    ).to(device)
    
    marigoldPipe.set_progress_bar_config(disable=True)

    return marigoldPipe

def depthMidasModel(): 
    # 1. Load the pre-trained MiDaS model and transforms
    # You can choose from different model types for varying speed/accuracy trade-offs
    model_type = "DPT_Large" # MiDaS v3 - Large
    # model_type = "DPT_Hybrid" # MiDaS v3 - Hybrid
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small

    # Load the model from PyTorch Hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)

    # Load the corresponding transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Set the device
    device = torch.device(setDevice())
    midas.to(device)
    midas.eval()  # Set the model to evaluation mode

    return midas, transform

def OneHotMask_Target(num_classes, img_size, target) -> torch.tensor:
    mask_np = np.array(target, dtype=np.int64) # Use int64 for class IDs
    one_hot_mask = np.zeros((num_classes, img_size[0], img_size[1]), dtype=np.float32)
    # print(f"one_hot_mask shape: {one_hot_mask.shape}")
    labels = list(range(num_classes))
    # Populate the one-hot tensor
    for i, class_id in enumerate(labels):                
        # print(f"idx - class_id: {i} - {class_id}")
        # print(f"mask_np == class_id: {np.where(mask_np == class_id)}")
        # print(one_hot_mask[i, :, :].shape)
        # For each class_id, set pixels in that channel to 1.0 where mask_np matches class_id
        one_hot_mask[i, :, :][mask_np == class_id] = 1.0

    one_hot_mask_tensor = torch.from_numpy(one_hot_mask)

    return one_hot_mask_tensor

def TransformImg(img):
    return (img*255).numpy().astype('uint8')

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_pseudo_depth_models=3, reduction=1):
        super().__init__()
        self.dim = dim
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.num_pseudo_depth_models = num_pseudo_depth_models

        num_channel = num_pseudo_depth_models * dim
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, groups=num_channel, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear((num_channel * 2), (num_channel * 2) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((num_channel * 2) // reduction, num_channel),
            nn.Sigmoid()
        )

    def forward(self, feats: torch.tensor):
        B, _, H, W = feats.size()
        # feats = torch.cat([feat1, feat2, feat3], dim=1)  # B, 3C, H, W

        feats = self.depthwise_conv(feats)

        avg_feat = self.pool_avg(feats).flatten(1)
        max_feat = self.pool_max(feats).flatten(1)
        attn = torch.cat([avg_feat, max_feat], dim=1)  # B, 6C
        attn = self.fc_layers(attn).view(B, self.num_pseudo_depth_models, self.dim, 1, 1)  # B, 3, C, 1, 1
        attn = attn.permute(1, 0, 2, 3, 4)  # 3, B, C, 1, 1

        return attn

class SpatialAttention(nn.Module):
    def __init__(self, dim, num_pseudo_depth_models=3, reduction=1):
        super().__init__()
        self.num_pseudo_depth_models = num_pseudo_depth_models
        self.conv_layers = nn.Sequential(
            nn.Conv2d(dim * num_pseudo_depth_models, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, num_pseudo_depth_models, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feats: torch.tensor):
        B, _, H, W = feats.size()
        # feats = torch.cat([feat1, feat2, feat3], dim=1)  # B, 3C, H, W
        attn_map = self.conv_layers(feats)
        attn_map = attn_map.view(B, self.num_pseudo_depth_models, 1, H, W).permute(1, 0, 2, 3, 4)  # 3, B, 1, H, W

        return attn_map

class PseudoDepthAggregationModule(nn.Module):
    def __init__(self, dim, num_pseudo_depth_models=3, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.dim = dim
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_attention = ChannelAttention(dim, num_pseudo_depth_models, reduction)
        self.spatial_attention = SpatialAttention(dim, num_pseudo_depth_models, reduction)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: tuple):
        
        featsMerge = torch.cat(feats, dim=1)  # B, 3C, H, W
        # print(featsMerge.shape)
        c_attn = self.channel_attention(featsMerge)
        s_attn = self.spatial_attention(featsMerge)

        assert type(feats) == tuple
        # assert len(feats) == self.dim
        assert len(c_attn) == len(s_attn)

        # out1 = feat1 + self.lambda_c * c_attn[0] * feat1 + self.lambda_s * s_attn[0] * feat1
        # out2 = feat2 + self.lambda_c * c_attn[1] * feat2 + self.lambda_s * s_attn[1] * feat2
        # out3 = feat3 + self.lambda_c * c_attn[2] * feat3 + self.lambda_s * s_attn[2] * feat3
        fused = 0
        for idx, featCur in enumerate(feats):
            cFeat_attn = self.lambda_c * c_attn[idx] * featCur
            sFeat_attn = self.lambda_s * s_attn[idx] * featCur
            out = featCur + cFeat_attn + sFeat_attn
            fused += out

        # fused = out1 + out2 + out3
        return fused

def add_depth_noise(depth_tensor, noise_factor=0.01):
    """Add relative Gaussian noise to depth tensor."""
    depth_tensor = depth_tensor.float()
    noise = torch.randn_like(depth_tensor) * depth_tensor * noise_factor
    return depth_tensor + noise

def Preprocess(rgb_np, depth_dict_np, mask_np):
    rgb = torch.from_numpy(rgb_np).permute(2,0,1).float() / 255.0
    depth_dict = {}
    for key, d in depth_dict_np.items():
        if d.dtype == np.ndarray:
            depth_dict[key] = torch.from_numpy(d).unsqueeze(0).float()    
    mask = torch.from_numpy(mask_np).long()

    return rgb, depth_dict, mask

def Augment(rgb: torch.tensor, depth_dict: dict[str, torch.tensor], mask: torch.tensor):
    """
    rgb: Tensor [3,H,W], float32 in [0,1]
    depth_dict: dict of depth tensors, e.g. {"raw": [1,H,W], "norm": [1,H,W], "hha": [3,H,W]}
    mask: Tensor [H,W] (long) or [C,H,W] (one-hot)
    """
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0

    # --- Geometric transforms (shared across all) ---
    if random.random() < 0.5:
        rgb = F.hflip(rgb)
        mask = F.hflip(mask)
        for k in depth_dict:
            depth_dict[k] = F.hflip(depth_dict[k])

    # if random.random() < 0.5:
    #     rgb = F.vflip(rgb)
    #     mask = F.vflip(mask)
    #     for k in depth_dict:
    #         depth_dict[k] = F.vflip(depth_dict[k])

    k = random.randint(0, 3)
    if k > 0:
        rgb = torch.rot90(rgb, k, [1, 2])
        mask = torch.rot90(mask, k, [0, 1]) if mask.ndim == 2 else torch.rot90(mask, k, [1, 2])
        for key, d in depth_dict.items():
            depth_dict[key] = torch.rot90(d, k, [0, 1]) if d.ndim == 2 else torch.rot90(d, k, [1, 2])

    # --- Photometric transforms (RGB only) ---
    if random.random() < 0.5:
        rgb = F.adjust_brightness(rgb, 1 + 0.1 * (random.random() - 0.5))
    if random.random() < 0.5:
        rgb = F.adjust_contrast(rgb, 1 + 0.1 * (random.random() - 0.5))
    if random.random() < 0.5:
        rgb = F.adjust_saturation(rgb, 1 + 0.1 * (random.random() - 0.5))
    if random.random() < 0.5:
        rgb = F.adjust_hue(rgb, 0.05 * (random.random() - 0.5))

    if random.random() < 0.3:
        rgb = F.gaussian_blur(rgb, kernel_size=(3, 3))

    # --- Noise injection ---
    if random.random() < 0.3:
        rgb = torch.clamp(rgb + torch.randn_like(rgb) * 0.02, 0, 1)
        for key in depth_dict:
            depth_dict[key] = add_depth_noise(depth_dict[key], noise_factor=0.01)

    return rgb, depth_dict, mask


def depth_to_hha_Dataloader(depth_map):
    depth_list = []
    for depth in depth_map:
        depth_list.append(depth_to_hha(depth))
    return torch.stack(depth_list, dim=0)

def depth_to_hha(depth_map, camera_matrix = np.array([[5.188579e+02, 0.000000e+00, 3.255824e+02],
                                                    [0.000000e+00, 5.194696e+02, 2.537362e+02],
                                                    [0.000000e+00, 0.000000e+00, 1.000000e+00]])):
    """
    Converts a 1-channel depth map tensor into a 3-channel HHA tensor on the GPU.
    
    Parameters:
    - depth_map: PyTorch Tensor of shape (H, W) or (1, H, W) on CUDA.
    - camera_matrix: 3x3 PyTorch Tensor or NumPy array.
    """
    # 1. Handle shapes and match device/dtype
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0) # Remove channel dim if (1, H, W)
        
    device = depth_map.device
    dtype = depth_map.dtype
    H, W = depth_map.shape
    
    if H < 2 or W < 2:
        raise ValueError(f"Depth map too small: {H}x{W}")

    K = torch.tensor(camera_matrix, device=device, dtype=dtype) if not isinstance(camera_matrix, torch.Tensor) else camera_matrix.to(device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 2. Generate 3D Point Cloud on GPU
    # indexing='xy' matches numpy meshgrid behavior
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype), 
                          torch.arange(W, device=device, dtype=dtype), indexing='ij')
    
    X = (x - cx) * depth_map / fx
    Y = (y - cy) * depth_map / fy
    Z = depth_map
    points_3d = torch.stack((X, Y, Z), dim=-1) # (H, W, 3)

    # 3. Gravity Vector (Default assumption)
    gravity_vector = torch.tensor([0, -1, 0], device=device, dtype=dtype)

    # 4. Channel 1: Disparity
    disparity = torch.where(Z > 0, 1.0 / Z, torch.zeros_like(Z))

    # 5. Channel 2: Height Above Ground
    height_raw = torch.sum(points_3d * gravity_vector, dim=-1)
    height = height_raw - torch.min(height_raw)

    # 6. Channel 3: Angle with Gravity (GPU Finite Differences)
    # PyTorch doesn't have np.gradient, so we use manual slicing for speed
    dz_dx = torch.zeros_like(Z)
    dz_dy = torch.zeros_like(Z)
    
    dz_dx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / 2.0
    dz_dy[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / 2.0
    
    normals = torch.stack((-dz_dx, -dz_dy, torch.ones_like(Z)), dim=-1)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    normals = torch.where(norm > 0, normals / norm, torch.zeros_like(normals))
    
    angle = torch.acos(torch.clamp(torch.sum(normals * gravity_vector, dim=-1), -1.0, 1.0))

    # 7. Normalize to 0.0 - 1.0 (Standard format for deep learning Tensors)
    def normalize_tensor(tensor):
        t_min, t_max = tensor.min(), tensor.max()
        if t_max - t_min > 0:
            return (tensor - t_min) / (t_max - t_min)
        return torch.zeros_like(tensor)

    hha_tensor = torch.stack((normalize_tensor(disparity), 
                              normalize_tensor(height), 
                              normalize_tensor(angle)), dim=0) # Shape: (3, H, W)
    
    return hha_tensor

