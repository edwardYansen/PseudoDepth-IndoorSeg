from module.DepthConstSet import DepthConstSet
from torch.utils.data import DataLoader, random_split
from .NyuDataset import PseudoDepthVisualization, GrayscalVisualization_IndividualChannels
from .DataTransform import PseudoDepthCreation, OneHotMask_Target, TransformsDepth, TransformImg, Augment, depth_to_hha
from torch.utils.data import Dataset
from .nyuv2_python_toolkit_master.nyuv2 import NYUv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # For visualization

class NYUv2_WithDepth(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 image_size=(256, 256),
                 transform_depth=False,
                 auto_augment=True,
                 transform_depth_dict = {DepthConstSet().marigold, DepthConstSet().midas, DepthConstSet().depthAnythingV2}):
        
        assert(split in ('train', 'test'))
        self.transform_depth = transform_depth
        self.auto_augment = auto_augment
        self.image_size = image_size
        self.transform_depth_dict = transform_depth_dict
        if transform_depth and (transform_depth_dict is None or transform_depth_dict == {}):
            self.transform_depth_dict = {DepthConstSet().marigold, DepthConstSet().midas, DepthConstSet().depthAnythingV2}

        self.common_transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor()
                                ])
        self.nyu_semantic13 = NYUv2( root=root, split=split, target_type='semantic', num_classes=13, 
                                transform=self.common_transform,
                                target_transform=transforms.Compose([
                                    transforms.Resize(image_size, interpolation=Image.NEAREST),
                                    transforms.Lambda(lambda lbl: torch.from_numpy( np.array(lbl, dtype='uint8')-1 ) ) # 0->255, 1->0, 2->1
                                ]),  
                            )
        
        self.nyu_depth = NYUv2( root=root, split=split, target_type='depth', 
                            transform=self.common_transform,
                            target_transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.Lambda(lambda lbl: torch.from_numpy( np.array(lbl, dtype='float') )/1e3 ) # uint16 to depth
                            ]),  
                        )
        
        if self.transform_depth:
            PDCreation = PseudoDepthCreation(self.transform_depth_dict)
            self.MarigoldPseudoDepth, self.MidasPseudoDepth, self.AnythingV2PseudoDepth = PDCreation.CreateListPseudoDepth(self.nyu_semantic13)
            
    def __getitem__(self, idx):
        # print(f"idx: {idx}")
        img, target = self.nyu_semantic13[idx]
        _, depth = self.nyu_depth[idx]

        # img = TransformImg(img)
        label = OneHotMask_Target(13, self.image_size, target)
        
        depthReturn = {}
        depthReturn[DepthConstSet().raw] = TransformsDepth(depth, self.common_transform)

        if self.transform_depth:
            if DepthConstSet().marigold in self.transform_depth_dict:
                depthReturn[DepthConstSet().marigold] = TransformsDepth(self.MarigoldPseudoDepth[idx], self.common_transform)
            if DepthConstSet().midas in self.transform_depth_dict:
                depthReturn[DepthConstSet().midas] = TransformsDepth(self.MidasPseudoDepth[idx], self.common_transform)
            if DepthConstSet().depthAnythingV2 in self.transform_depth_dict:
                depthReturn[DepthConstSet().depthAnythingV2] = TransformsDepth(self.AnythingV2PseudoDepth[idx], self.common_transform)

        if not self.auto_augment:
            return (img, depthReturn, label)
        
        imageAugmented, depthAugmented, maskAugmented = Augment(img, depthReturn, label)
        return (imageAugmented, depthAugmented, maskAugmented)

    def __len__(self):
        return len(self.nyu_semantic13)

def get_train_val_dataloaders(dataset_path, batch_size=4, num_workers=0, image_size=(256, 256), transform_depth=False, auto_augment=True, transform_depth_dict = {}):  
    if transform_depth and (transform_depth_dict is None or transform_depth_dict == {}):
        transform_depth_dict = {DepthConstSet().marigold, DepthConstSet().midas, DepthConstSet().depthAnythingV2}
    train_full_dataset = NYUv2_WithDepth(root=dataset_path, split="train", image_size=image_size, transform_depth=transform_depth, auto_augment=auto_augment, transform_depth_dict=transform_depth_dict)
    test_dataset = NYUv2_WithDepth(root=dataset_path, split="test", image_size=image_size, transform_depth=transform_depth, auto_augment=auto_augment, transform_depth_dict=transform_depth_dict)

    dataset_size = len(train_full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size_split = dataset_size - train_size
    train_dataset, val_dataset = random_split(train_full_dataset, [train_size, val_size_split])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data for better generalization
        num_workers=num_workers,
        pin_memory=True # Speeds up data transfer to GPU
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded. Total samples: {dataset_size}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataloader)}, Test samples: {len(test_dataloader)}")
    # print(f"Number of classes for segmentation (including background): {num_classes}")

    return train_dataloader, val_dataloader, test_dataloader

def visualize_batch(batch, num_classes, totalShow=3, transform_depth=False, transform_hha=False):
    """
    Visualizes a batch of RGB images, depth maps, and segmentation labels.

    Args:
        batch (tuple): A batch from the DataLoader, containing (rgb_images, (depth_maps, labels)).
        num_classes (int): The total number of classes, used for colormap.
    """
    
    (rgb_images, depth_maps, labels) = batch
    
    # depth_maps = targets[0]
    # labels = targets[1] # Labels are now one-hot: (N, C, H, W)
    # print(depth_maps)

    batch_size = rgb_images.shape[0]
    if batch_size > totalShow:
        batch_size = totalShow


    # Denormalize RGB images if normalization was applied
    # rgb_images = DenormalizeRGBimages(rgb_images, dataset_instance)

    for i in range(batch_size):
        # Convert tensors to NumPy arrays for plotting
        # visualize_batch_single(rgb_images[i], depth_maps[i], labels[i], i, num_classes, dataset_instance)
        rgb_img_np = rgb_images[i].permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C
        
        # Convert one-hot label back to class indices for visualization
        # This is the key step for visualizing (N, 14, 480, 640) labels
        # label_np = torch.argmax(labels[i]).cpu().numpy() # C, H, W -> H, W
        label_np = labels[i]
                
        label_map = torch.argmax(label_np, dim=0).cpu()
        label_numpy = label_map.numpy()
        
        # Denormalize depth map for better visualization (scale back to original range)
        # We can use a colormap for depth for better visibility
        
        plt.figure(figsize=(18, 6)) # Adjust figure size dynamically
        plt.suptitle(f'Visualization for Sample {i + 1}', fontsize=16)

        # Plot RGB Image
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_img_np)
        plt.title(f'RGB Image')
        plt.axis('off')

        # # Plot Depth Map
        plt.subplot(1, 3, 2)
        # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
        depth_map_np = depth_maps[DepthConstSet().raw][i].squeeze(0).cpu().numpy() # 1, H, W -> H, W
        if transform_hha:
            depth_hha = depth_to_hha(depth_maps[DepthConstSet().raw][i])
            depth_map_np = depth_hha.permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C
        plt.imshow(depth_map_np, cmap='viridis', vmin=0, vmax=np.max(depth_map_np)) 
        plt.title(f'Depth Map')
        plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
        plt.axis('off')

        # Predicted Segmentation
        plt.subplot(1, 3, 3)
        plt.imshow(label_numpy, cmap='tab20', vmin=0, vmax=num_classes - 1)
        plt.title(f'Predicted Segmentation')
        plt.colorbar(ticks=range(num_classes), fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        if transform_depth:
            PseudoDepthVisualization(depth_maps[DepthConstSet().marigold][i], depth_maps[DepthConstSet().midas][i], depth_maps[DepthConstSet().depthAnythingV2][i])
        # visualize_depth_single(rgb_img_np, depth_map_np)
        GrayscalVisualization_IndividualChannels(num_classes, label_np, None)