import h5py
import matplotlib.pyplot as plt # For visualization
from .DepthConstSet import DepthConstSet
import numpy as np
import sys
import torch
import cv2
from PIL import Image # Import PIL for Image.fromarray
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
from diffusers import MarigoldDepthPipeline, DiffusionPipeline
from .DataTransform import Transform_data_depth

def Transform_data(images, depths, labels, transform=None, normalize_rgb=None, depth_norm=10, topK_labels=None, num_classes=None, img_size=(480, 640)):
    """
    A placeholder class to define the type hint for the transform parameter.
    This class itself doesn't contain any logic but serves for clarity.
    """
    # image = cv2.rotate(images.transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE) # RGB image (H, W, C)
    # depth = cv2.rotate(depths, cv2.ROTATE_90_CLOCKWISE) # Depth map (H, W)
    # one_hot_mask = cv2.rotate(labels, cv2.ROTATE_90_CLOCKWISE).astype(np.float32) # Semantic label (H, W)
    image = images
    depth = depths
    one_hot_mask = labels
    # print(image.shape)
    # print(depth.shape)
    # print(one_hot_mask.shape)
    # print(img_size)
    # Apply transformations consistently to image, depth, and label
    if transform:
        # Convert numpy arrays to PIL Images for torchvision transforms
        image_pil = Image.fromarray(image)
        
        # For depth, scale to 0-255 for PIL conversion, then convert back after ToTensor
        depth_scaled_for_pil = np.clip(depth / depth_norm * 255.0, 0, 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_scaled_for_pil, mode='L') 
        
        # For label, convert to PIL Image (mode 'L' for grayscale)
        # Note: `label` here holds the class IDs (0 to num_classes-1)
        label_pil = Image.fromarray(one_hot_mask.astype(np.uint8), mode='L') 

        # Store random state before applying first transform to ensure consistency
        state = torch.get_rng_state() 
        image = transform(image_pil) # Apply transform to RGB image (PIL -> Tensor)

        # Reset random state and apply same transforms to depth and label
        torch.set_rng_state(state) 
        depth = transform(depth_pil) # Apply transform to depth (PIL -> Tensor)
        depth = (depth / 255.0) * depth_norm # Scale depth back to original range

        torch.set_rng_state(state) 
        # Apply transform to label. It will be resized to self.img_size.
        # Then convert to numpy and cast to int64 for proper class ID handling.
        # transformed_label_pil = self.transform(label_pil) # This converts to Tensor (1, H, W)
        # The next step converts this tensor back to numpy, then extracts the 
        # single channel and ensures it's Long (int64 for numpy).
        # mask_np = transformed_label_pil.squeeze(0).cpu().numpy().astype(np.int64) 
        mask=F.resize(label_pil, size = img_size, interpolation=transforms.InterpolationMode.NEAREST)
        # mask_np = np.array(mask.unsqueeze(0).cpu(), dtype=np.int64) # Use int64 for class IDs
        mask_np = np.array(mask, dtype=np.int64) # Use int64 for class IDs
        # mask_np = mask.squeeze(0).cpu().numpy().astype(np.int64) # Use int64 for class IDs

        # print(f"mask_np: {np.unique(mask_np)}")
        # print(f"label_pil: {type(label_pil)}")
        # print(f"mask: {type(mask)}")
        # print(f"mask_np shape: {type(mask_np)}")
        # print(f"mask shape: {mask.shape}")
        # print(f"mask_np shape: {mask_np.shape}")
        
        if topK_labels:
            if len(topK_labels) != num_classes:
                print(f"Error topK_labels != num_classes: {len(topK_labels)} != {num_classes}")
                sys.exit(1)
                            
            # Initialize an empty one-hot tensor (C, H, W)
            one_hot_mask = np.zeros((num_classes, img_size[0], img_size[1]), dtype=np.float32)
            # print(f"one_hot_mask shape: {one_hot_mask.shape}")
            
            # Populate the one-hot tensor
            for i, class_id in enumerate(topK_labels):                
                # print(f"idx - class_id: {i} - {class_id}")
                # print(f"mask_np == class_id: {np.where(mask_np == class_id)}")
                # print(one_hot_mask[i, :, :].shape)
                # For each class_id, set pixels in that channel to 1.0 where mask_np matches class_id
                one_hot_mask[i, :, :][mask_np == class_id] = 1.0
        
        # Convert the numpy one_hot_mask back to a torch tensor
        # one_hot_mask = torch.from_numpy(one_hot_mask)

        # Ensure depth map has a channel dimension (1, H, W)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

    # Apply ImageNet normalization to RGB image if specified
    if normalize_rgb:
        image = normalize_rgb(image)
    
    return (image, depth, one_hot_mask)

def transformsDepth(depth, transform, depth_norm=10):
    # depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE) # Depth map (H, W)
    
    if not transform: return depth
    
    # For depth, scale to 0-255 for PIL conversion, then convert back after ToTensor
    depth_scaled_for_pil = np.clip(depth / depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_scaled_for_pil, mode='L') 

    # Reset random state and apply same transforms to depth and label
    depth = transform(depth_pil) # Apply transform to depth (PIL -> Tensor)
    depth = (depth / 255.0) * depth_norm # Scale depth back to original range
    
    # Ensure depth map has a channel dimension (1, H, W)
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    
    return depth    

# --- 1. NYU Depth Dataset Class ---
class NyuDataset(Dataset):
    def __init__(self, root, topk_labels=None, transform=None, normalize=False, depth_norm=10, num_classes=None, img_size=(480, 640), transform_depth=False):
        """
            NYU Depth Dataset Loader.

            Args:
                root (str): Path to NYU dataset .mat file (e.g., 'nyu_depth_v2_labeled.mat').
                topk_labels (list, optional): A list of original label IDs to keep. Other labels will be
                                              mapped to 0 (unlabeled/background). If None, all labels
                                              will be considered up to `num_classes`.
                transform (torchvision.transforms.Compose, optional): Transforms to apply to RGB image,
                                                                      depth map, and segmentation label.
                                                                      Geometric transforms will be applied
                                                                      consistently across all three.
                normalize (bool): If True, applies ImageNet normalization to RGB images.
                depth_norm (float): Normalization factor for depth maps. Depth values will be divided by this.
                num_classes (int): The total number of output classes for segmentation, including the
                                   unlabeled/background class (label 0). This is crucial for model output
                                   and metric calculation. Must be provided.
                img_size (tuple): The target (height, width) for resizing images and masks.
        """
        if num_classes is None:
            raise ValueError("`num_classes` must be provided to NyuDataset to define the segmentation output space.")

        self.topk_labels = topk_labels
        self.transform = transform
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform_depth = transform_depth

        # ImageNet normalization for RGB images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize_rgb = transforms.Normalize(mean, std) if normalize else None
        # Store mean and std for denormalization during visualization
        self.mean_rgb = torch.tensor(mean).view(3, 1, 1) if normalize else None
        self.std_rgb = torch.tensor(std).view(3, 1, 1) if normalize else None

        # Normalization factor for depth maps
        self.depth_norm = depth_norm

        # Open .mat file as an h5 object
        try:
            self.h5_obj = h5py.File(root, mode='r')
            print(self.h5_obj.keys())
        except Exception as e:
            print(f"Error opening HDF5 file at {root}: {e}")
            print("Please ensure the dataset file 'nyu_depth_v2_labeled.mat' is correctly placed and accessible.")
            sys.exit(1) # Exit if the dataset file cannot be opened

        # Obtain desired groups from the HDF5 file
        self.images = self.h5_obj['images'] # RGB images (C, H, W)
        print(type(self.images))
        self.depths = self.h5_obj['depths'] # Depth maps (H, W)
        self.labels = self.h5_obj['labels'] # Semantic class mask for each image (H, W)
        self.names = self.h5_obj['names']   # Semantic class labels (for `str_label`)
        if(self.transform_depth):
            self.depths1, self.depths2, self.depths3 = Transform_data_depth(self.images)
        # print(self.names.shape)
        # print(self.names)

        # Initialize the map for string names of our *mapped* classes (0 to num_classes - 1)
        self._mapped_class_string_names = {}
        self._mapped_class_string_names[0] = 'unlabeled' # Class 0 is always background/unlabeled

        # If topk_labels are specified, create a mapping from original labels to new sequential labels.
        # Label 0 is always reserved for background/unlabeled.
        if self.topk_labels:
            self.original_to_new_label_map = {0: 0} # 0 remains unlabeled
            # Sort for consistent mapping to ensure stable class ID assignments
            for i, original_lbl in enumerate(sorted(self.topk_labels)):
                self.original_to_new_label_map[original_lbl] = i # New labels start from 1
            print(self.original_to_new_label_map)

            # Ensure num_classes matches the new mapping for background
            if self.num_classes != len(self.topk_labels):
                print(f"Warning: `num_classes` ({self.num_classes}) does not match "
                      f"len(topk_labels) + 1 ({len(self.topk_labels)}). "
                      f"Using `len(topk_labels) + 1` as the effective number of classes.")
                self.num_classes = len(self.topk_labels)
            
            # Populate _mapped_class_string_names using the new mapping
            for original_lbl_id, mapped_id in self.original_to_new_label_map.items():
                if mapped_id != 0: # Already handled background (mapped_id=0)
                    try:
                        # Fetch original name from the h5_obj's 'names' dataset
                        # The 'names' dataset is 1-indexed in NYU for class labels (original_lbl_id),
                        # but 0-indexed in array access (original_lbl_id - 1).
                        original_name_bytes = self.h5_obj[self.names[0, original_lbl_id-1]]
                        original_name = ''.join(chr(i[0]) for i in original_name_bytes)
                        print(f"mapped classes: {original_lbl_id} -> {mapped_id} - {original_name}")
                        self._mapped_class_string_names[mapped_id] = original_name
                    except Exception:
                        self._mapped_class_string_names[mapped_id] = f'mapped_class_{mapped_id}_(orig_{original_lbl_id})'
        else:
            # If no topk_labels, assume labels are already within 0 to num_classes-1 range.
            # Directly map original 1-indexed NYU names to our 0-indexed internal classes.
            # Iterate through original NYU names (which are 1-indexed in the dataset for class labels)
            for original_idx_in_names_array in range(self.names.shape[1]):
                original_nyu_label_id = original_idx_in_names_array + 1 # Corresponding 1-indexed NYU label ID
                # Only map if this original ID is within our `num_classes` range (adjusted to 0-indexed)
                # if original_nyu_label_id < self.num_classes: 
                try:
                    original_name_bytes = self.h5_obj[self.names[0, original_idx_in_names_array]]
                    original_name = ''.join(chr(i[0]) for i in original_name_bytes)
                    self._mapped_class_string_names[original_nyu_label_id] = original_name 
                except Exception:
                    self._mapped_class_string_names[original_nyu_label_id] = f'original_class_{original_nyu_label_id}'

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getLabelList__(self):
        for x in self._mapped_class_string_names:
            print(f"{x}: {self._mapped_class_string_names[x]}")


    def __getitem__(self, idx):
        """
            Retrieves an RGB image, depth map, and corresponding semantic label.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: (img, (depth1, depth2, depth3), GT)
                    img (torch.Tensor): Transformed and normalized RGB image (C, H, W).
                    depth1-3 (torch.Tensor): Transformed and normalized depth map (1, H, W).
                    GT (torch.Tensor): Transformed and one-hot encoded semantic segmentation label (num_classes, H, W), FloatTensor.
        """
        # print(f"idx: {idx}")
        image = cv2.rotate(self.images[idx].transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE) # RGB image (H, W, C)
        depth = cv2.rotate(self.depths[idx], cv2.ROTATE_90_CLOCKWISE) # Depth map (H, W)
        one_hot_mask = cv2.rotate(self.labels[idx], cv2.ROTATE_90_CLOCKWISE).astype(np.float32) # Semantic label (H, W)
        # print(type(image))


        (img, depthRaw, GT) = Transform_data(image, depth, one_hot_mask
                              , transform=self.transform
                              , normalize_rgb=self.normalize_rgb
                              , depth_norm=self.depth_norm
                              , topK_labels=self.topk_labels
                              , num_classes=self.num_classes
                              , img_size=self.img_size)
        depthReturn = {}
        depthReturn[DepthConstSet().raw]=depthRaw

        if self.transform_depth:
            depthMarigold = transformsDepth(self.depths1[idx], self.transform, self.depth_norm)
            depthMidas = transformsDepth(self.depths2[idx], self.transform, self.depth_norm)
            depthAnythingV2 = transformsDepth(self.depths3[idx], self.transform, self.depth_norm)
            depthReturn[DepthConstSet().marigold]=depthMarigold
            depthReturn[DepthConstSet().midas]=depthMidas
            depthReturn[DepthConstSet().depthAnythingV2]=depthAnythingV2

        # return (img, depthRaw, GT)
        return (img, depthReturn, GT)


    def str_label(self, mapped_class_id):
        """ 
            Obtains string label for a *mapped* class index.
            This function now directly uses the pre-computed map `_mapped_class_string_names`.
            
            Args:
                mapped_class_id (int): The 0-indexed class ID (from 0 to num_classes-1).

            Returns:
                str: The string name of the class.
        """
        return self._mapped_class_string_names.get(mapped_class_id, f'unknown_mapped_class_{mapped_class_id}')

    def close(self):
        """Closes the HDF5 file handle."""
        if self.h5_obj:
            self.h5_obj.close()

    def __exit__(self, *args):
        """Ensures the HDF5 file is closed when exiting a 'with' statement."""
        self.close()

def convertNpToImg(img:np.ndarray):
    img = Image.fromarray(img)
    return img

def depthAnything_create(img: np.ndarray):    
    # 2. Load the model and its processor
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Note: The model you initialized with a configuration above
    # does not have the pre-trained weights. You need to load them
    # using from_pretrained()
    model = DepthAnythingForDepthEstimation.from_pretrained(model_name)

    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3. Pre-process the input image and perform inference
    # input_image = Image.fromarray(img)
    input_image = img
    if (type(img) is not torch.Tensor):
        input_image = convertNpToImg(img)

    inputs = image_processor(images=input_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 4. Post-process the output
    # Resize the predicted depth map to the original image's size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output_numpy = prediction.squeeze().cpu().numpy()

    # Normalize the numpy array to the range [0, 1]
    normalized_output = (output_numpy - output_numpy.min()) / (output_numpy.max() - output_numpy.min())

    # *** INVERT THE DEPTH VALUES HERE ***
    # This makes 0 closest and 1 farthest
    inverted_output = 1 - normalized_output

    return inverted_output

def depthMarigold_create(img: np.ndarray):    
    # 2. Convert the NumPy array to a PIL Image.
    # The Marigold pipeline expects a PIL Image object as input.
    # input_image = Image.fromarray(img.astype(np.uint8))
    input_image = img
    if (type(img) is not torch.Tensor):
        input_image = convertNpToImg(img)

    # 3. Load the Marigold pipeline.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    marigoldPipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        source="local"
    ).to(device)

    # 4. Perform inference with the PIL Image.
    # The pipeline handles all the heavy lifting, including preprocessing.
    result = marigoldPipe(
        input_image,
        # ensemble_size=10,
        # num_inference_steps=20,
        # show_progress_bar=True,
        batch_size=8,
        output_type = "np"
    )

    depth_map_numpy = result.prediction.squeeze()

    return depth_map_numpy

def depthMidas_create(img: np.ndarray): 
    # input_image = convertNpToImg(img)
    
    # 1. Load the pre-trained MiDaS model and transforms
    # You can choose from different model types for varying speed/accuracy trade-offs
    model_type = "DPT_Large" # MiDaS v3 - Large
    # model_type = "DPT_Hybrid" # MiDaS v3 - Hybrid
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small
    if (type(img) is torch.Tensor):
        img = img.numpy()

    # Load the model from PyTorch Hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type, source = "local", pretrained=True)

    # Load the corresponding transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",source = "local" ,pretrained=True)
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Set the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()  # Set the model to evaluation mode

    # Read the image using OpenCV and apply transforms
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # 3. Predict the depth
    with torch.no_grad():
        prediction = midas(input_batch)

    # 4. Post-process and visualize the output
    # Resize the prediction to match the original image size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Convert the tensor to a NumPy array
    # output = prediction.cpu().numpy()
    output_numpy = prediction.squeeze().cpu().numpy()

    # Normalize the numpy array to the range [0, 1]
    normalized_output = (output_numpy - output_numpy.min()) / (output_numpy.max() - output_numpy.min())

    # *** INVERT THE DEPTH VALUES HERE ***
    # This makes 0 closest and 1 farthest
    inverted_output = 1 - normalized_output

    return inverted_output

def create_pseudo_depth(img: np.ndarray):
    depthMarigold = depthMarigold_create(img)
    depthMidas = depthMidas_create(img)
    depthAnything = depthAnything_create(img)

    return (depthMarigold, depthMidas, depthAnything)

def visualize_depth_single(img, depth):
    (depthMarigold, depthMidas, depthAnything) = create_pseudo_depth(img)

    plt.figure(figsize=(18, 6)) # Adjust figure size dynamically
    
    plt.subplot(1, 4, 1)
    # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
    plt.imshow(depth, cmap='viridis', vmin=0, vmax=np.max(depth)) 
    plt.title(f'Depth raw Map')
    plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
    plt.axis('off')

    plt.subplot(1, 4, 2)
    # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
    plt.imshow(depthMarigold, cmap='viridis', vmin=0, vmax=np.max(depthMarigold)) 
    plt.title(f'Depth Marigold Map')
    plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
    plt.axis('off')

    plt.subplot(1, 4, 3)
    # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
    plt.imshow(depthMidas, cmap='viridis', vmin=0, vmax=np.max(depthMidas)) 
    plt.title(f'Depth Midas Map')
    plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
    plt.imshow(depthAnything, cmap='viridis', vmin=0, vmax=np.max(depthAnything)) 
    plt.title(f'Depth Anythingv2 Map')
    plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
    plt.axis('off')
    plt.show()

# --- 3. Dataloader and Splitting Function ---
def get_train_val_dataloaders(dataset_path, train_ratio=0.8, batch_size=4, num_workers=0,
                              topk_labels=None, image_size=(256, 256), num_classes=None, transform_depth=False):
    """
    Creates train and validation DataLoaders from the NYU dataset.

    Args:
        dataset_path (str): Path to the NYU dataset .mat file.
        train_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80% train, 20% val).
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading. 0 means data is loaded
                           in the main process (good for debugging, slower for large datasets).
        topk_labels (list, optional): List of original label IDs to consider.
        image_size (tuple): Target size for images and masks (H, W). All data will be resized to this.
        num_classes (int): Number of output classes for segmentation. This is a crucial parameter
                           and must be provided.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    if num_classes is None:
        raise ValueError("`num_classes` must be provided to `get_train_val_dataloaders`.")

    # Define common transforms for both training and validation sets.
    # For training, you might add more aggressive augmentations (e.g., RandomHorizontalFlip,
    # RandomRotation, ColorJitter) *before* Resize and ToTensor.
    # Ensure `interpolation=transforms.InterpolationMode.NEAREST` for labels to preserve class IDs.
    common_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.ToTensor() # Converts PIL Image to Tensor (H, W, C) to (C, H, W) and normalizes to 0-1
    ])

    # Initialize the full dataset
    full_dataset = NyuDataset(
        root=dataset_path,
        topk_labels=topk_labels,
        transform=common_transforms,
        normalize=True, # Apply ImageNet normalization for RGB
        depth_norm=10,  # Normalization factor for depth (adjust based on your data's depth range)
        num_classes=num_classes # Pass num_classes to the dataset
        ,img_size=image_size # Pass image_size to the dataset
        ,transform_depth=transform_depth
    )

    # Split the dataset into training and validation sets
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size_split = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size_split])
    val_size = int(val_size_split * 0.5)
    test_size = val_size_split-val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

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

def PseudoDepthVisualization(depth1, depth2, depth3):
    # --- Strategy 1: Grayscale Visualization of Individual Channels ---
    plt.figure(figsize=(18, 10))
    plt.suptitle("Pseudo Depth Visualization", fontsize=16)
    
    plt.subplot(1, 3, 1)
    depth1_map_np = depth1.squeeze(0).cpu().numpy() # 1, H, W -> H, W
    plt.imshow(depth1_map_np)
    plt.title(f'Sample - Depth Marigold')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    depth2_map_np = depth2.squeeze(0).cpu().numpy() # 1, H, W -> H, W
    plt.imshow(depth2_map_np)
    plt.title(f'Sample - Depth Midas')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    depth3_map_np = depth3.squeeze(0).cpu().numpy() # 1, H, W -> H, W
    plt.imshow(depth3_map_np)
    plt.title(f'Sample - Depth Anythingv2')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def GrayscalVisualization_IndividualChannels(num_channels, segmentation_output, batch=None):
    # --- Strategy 1: Grayscale Visualization of Individual Channels ---
    plt.figure(figsize=(18, 10))
    plt.suptitle("Individual Grayscale Channels", fontsize=16)
    for i in range(num_channels):
        plt.subplot(3, 5, i + 1) # Adjust subplot grid as needed
        channel = segmentation_output[i,:, :]        
        print(f"Channel {i}: {channel.max()}")

        plt.imshow(channel, cmap='gray')
        if not batch:
            plt.title(f'Channel {(i)}')
        else:
            plt.title(f'Channel {batch.str_label(i)}')
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_batch_single(images, depths, label_np, idx, num_classes, dataset_instance):
    # rgb_images = DenormalizeRGBimages(images, dataset_instance)
    rgb_img_np = images.permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C
    depth_map_np = depths.squeeze(0).cpu().numpy() # 1, H, W -> H, W
                
    label_tensor = torch.from_numpy(label_np)
    label_map = torch.argmax(label_tensor, dim=0).cpu()
    # label_map = torch.argmax(label_np, dim=0).cpu()
    label_numpy = label_map.numpy()
    
    # Denormalize depth map for better visualization (scale back to original range)
    # We can use a colormap for depth for better visibility
    
    plt.figure(figsize=(18, 6)) # Adjust figure size dynamically

    # Plot RGB Image
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img_np)
    plt.title(f'Sample {idx} - RGB Image')
    plt.axis('off')

    # Plot Depth Map
    plt.subplot(1, 3, 2)
    # Use 'viridis' colormap for depth, ensure values are scaled for good contrast
    plt.imshow(depth_map_np, cmap='viridis', vmin=0, vmax=np.max(depth_map_np)) 
    plt.title(f'Sample {idx} - Depth Map')
    plt.colorbar(fraction=0.046, pad=0.04) # Add color bar for depth
    plt.axis('off')

    # Predicted Segmentation
    plt.subplot(1, 3, 3)
    plt.imshow(label_numpy, cmap='tab20', vmin=0, vmax=num_classes - 1)
    plt.title(f'Sample {idx} - Predicted Segmentation')
    plt.colorbar(ticks=range(num_classes), fraction=0.046, pad=0.04)
    plt.axis('off')

    
    GrayscalVisualization_IndividualChannels(num_classes, label_np, dataset_instance)
    plt.suptitle(f'Visualization for Sample {idx}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def DenormalizeRGBimages(rgb_images, dataset_instance):    
    # Denormalize RGB images if normalization was applied
    if dataset_instance.normalize_rgb:
        # Clone to avoid modifying the original tensor
        rgb_images_denorm = rgb_images.clone()
        for i in range(3):
            rgb_images_denorm[:, i, :, :] = rgb_images_denorm[:, i, :, :] * dataset_instance.std_rgb[i] + dataset_instance.mean_rgb[i]
        rgb_images = rgb_images_denorm.clamp(0, 1) # Clamp to [0, 1] range
    
    return rgb_images


# --- 5. Visualization Function ---
def visualize_batch(batch, num_classes, depth_norm, dataset_instance, totalShow=3, transform_depth=False):
    """
    Visualizes a batch of RGB images, depth maps, and segmentation labels.

    Args:
        batch (tuple): A batch from the DataLoader, containing (rgb_images, (depth_maps, labels)).
        num_classes (int): The total number of classes, used for colormap.
        depth_norm (float): The normalization factor used for depth, to denormalize for visualization.
        dataset_instance (NyuDataset): An instance of the NyuDataset to access mean/std for denormalization.
    """
    
    (rgb_images, depth_maps, labels) = batch
    
    # depth_maps = targets[0]
    # labels = targets[1] # Labels are now one-hot: (N, C, H, W)
    # print(depth_maps)

    batch_size = rgb_images.shape[0]
    if batch_size > totalShow:
        batch_size = totalShow


    # Denormalize RGB images if normalization was applied
    rgb_images = DenormalizeRGBimages(rgb_images, dataset_instance)

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
        GrayscalVisualization_IndividualChannels(num_classes, label_np, dataset_instance)

from .utils import *
def showData(idx, batch, topK_labels, dataset):
    # random_indices = np.random.choice(len(dataset), show).tolist()

    rgb_images, (depth_maps, labels) = batch

    # for i in range(BATCH_SIZE):
    # image, (depth, label) = dataset[i]
    # image_np, depth_np, label_np = convert_to_numpy(image, depth, label)
    plt.figure(figsize=(18, 6)) # Adjust figure size dynamically
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_images)
    plt.title(f'RGB Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(depth_maps, cmap='nipy_spectral')
    plt.title(f'Depth Image')
    plt.axis('off')
    
    label_range = np.unique(labels)
    maxLabel = label_range[-1]
    print(f"Label range: {label_range}")
    plt.subplot(1, 3, 3)
    plt.imshow(labels, cmap='tab20', vmin=0, vmax=maxLabel - 1)
    plt.title(f'Label Image')
    plt.colorbar(ticks=label_range, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(f'Visualization for Original Data {idx}', fontsize=16)        
    plt.show()

    showDataTransform(rgb_images, depth_maps, labels, idx, topK_labels=topK_labels, batch=dataset)

# global IMAGE_SIZE
# # IMAGE_SIZE = (480, 480)
# global NUM_CLASSES

def setImgSize(size):
    global IMAGE_SIZE
    IMAGE_SIZE = size

def getImgSize():
    return IMAGE_SIZE

def setNumClasses(num):
    global NUM_CLASSES
    NUM_CLASSES = num

def getNumClasses():
    return NUM_CLASSES


def showDataTransform(images, depths, labels, idx, topK_labels=None, batch=None):    
    common_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.ToTensor() # Converts PIL Image to Tensor (H, W, C) to (C, H, W) and normalizes to 0-1
    ])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_rgb = transforms.Normalize(mean, std)
    image, (depth, label) = Transform_data(images, depths, labels, 
                                           transform=common_transforms, 
                                           normalize_rgb=normalize_rgb, 
                                           depth_norm=10, 
                                           topK_labels=topK_labels, 
                                           num_classes=NUM_CLASSES, 
                                           img_size=IMAGE_SIZE)


    plt.figure(figsize=(18, 6)) # Adjust figure size dynamically
    rgb_img_np = image.permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img_np)
    plt.title(f'RGB Image')
    plt.axis('off')
    
    depth_map_np = depth.squeeze(0).cpu().numpy() # 1, H, W -> H, W
    plt.subplot(1, 3, 2)
    plt.imshow(depth_map_np, cmap='nipy_spectral')
    plt.title(f'Depth Image')
    plt.axis('off')
    
    label_tensor = torch.from_numpy(label)
    label_map = torch.argmax(label_tensor, dim=0).cpu()
    label_numpy = label_map.numpy()
    plt.subplot(1, 3, 3)
    plt.imshow(label_numpy, cmap='tab20', vmin=0, vmax=len(topK_labels))
    plt.title(f'Label Image')
    plt.colorbar(ticks=topK_labels, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(f'Visualization for Transform Data {idx}', fontsize=16)
    # GrayscalVisualization_IndividualChannels(len(topK_labels), label, batch=batch)
    plt.show()