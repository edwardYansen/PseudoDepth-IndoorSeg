import torchmetrics # Import torchmetrics for metrics like Accuracy and JaccardIndex (mIoU)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss # Or other suitable loss for segmentation
import torch.optim as optim
import matplotlib.pyplot as plt # For visualization
import os
import numpy as np
# import mlflow
# from mlflow.models import infer_signature
from module.DepthConstSet import DepthConstSet
from module.DataTransform import PseudoDepthAggregationModule
from .DataTransform import depth_to_hha_Dataloader


# --- 4. Training Code ---
def calculate_accuracy(predictions, targets, BATCH_SIZE):
    """
    Calculates pixel-wise accuracy for semantic segmentation.

    Args:
        predictions (torch.Tensor): Model predictions (logits), shape (N, C, H, W).
        targets (torch.Tensor): Ground truth labels (one-hot), shape (N, C, H, W).

    Returns:
        float: Pixel accuracy (0.0 to 1.0).
    """
    # Get the class with the highest probability for each pixel from predictions
    # print(f"predictions labels: {predictions.shape[0]}")
    if(predictions.shape[0]==BATCH_SIZE):
        _, predicted_labels = torch.argmax(predictions, 1) # Shape (N, H, W)
    else:
        predicted_labels = predictions.unsqueeze(0) # Shape (N, H, W)
    
    # Get the class with the highest probability for each pixel from one-hot targets
    # print(f"true labels: {targets.shape[0]}")    
    if(targets.shape[0]==BATCH_SIZE):
        _, true_labels = torch.argmax(targets, 1) # Shape (N, H, W)
    else:
        true_labels = targets.unsqueeze(0) # Shape (N, H, W)


    # Compare predicted labels with ground truth labels
    correct_pixels = (predicted_labels == true_labels).sum().item()
    total_pixels = true_labels.numel() # Total number of elements (pixels) in the target tensor
    
    accuracy = correct_pixels / total_pixels
    return accuracy

def calculate_miou(predictions, targets, num_classes, BATCH_SIZE):
    """
    Calculates Mean Intersection over Union (mIoU) for semantic segmentation.
    This metric is more robust than pixel accuracy for imbalanced classes.

    Args:
        predictions (torch.Tensor): Model predictions (logits), shape (N, C, H, W).
        targets (torch.Tensor): Ground truth labels (one-hot), shape (N, C, H, W).
        num_classes (int): Total number of classes, including background/unlabeled.

    Returns:
        float: Mean IoU (0.0 to 1.0). Returns 0.0 if no classes are found in the batch.
    """
    # print(f"predicted labels: {predictions.shape}")
    if(predictions.shape[0]==BATCH_SIZE):
        _, predicted_labels = torch.argmax(predictions, 1) # Shape (N, H, W)
    else:
        predicted_labels = predictions.unsqueeze(0) # Shape (N, H, W)
    # print(f"true labels: {targets.shape}")
    if(targets.shape[0]==BATCH_SIZE):
        _, true_labels = torch.argmax(targets, 1) # Shape (N, H, W)
    else:
        true_labels = targets.unsqueeze(0) # Shape (N, H, W)


    iou_per_class = []
    for cls in range(num_classes):
        # Create binary masks for the current class for both prediction and target
        pred_mask = (predicted_labels == cls)
        target_mask = (true_labels == cls)

        # Calculate intersection and union
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            # If the current class is not present in either the prediction or the target
            # for this batch, its IoU is undefined. We skip it to avoid NaN and
            # not penalize for classes absent in the current batch.
            continue
        
        iou = intersection / union
        iou_per_class.append(iou)
    
    if len(iou_per_class) == 0:
        # If no classes were found in the current batch (e.g., very small batch size
        # or specific filtering), return 0.0 to avoid division by zero.
        return 0.0 

    return np.mean(iou_per_class)

def get_listDepth(depthDict: dict, device, pseudo_depth_models: tuple = [DepthConstSet().marigold,DepthConstSet().midas,DepthConstSet().depthAnythingV2]) -> tuple:
    listDepth = []
    for depthModel in pseudo_depth_models:
        listDepth.append(depthDict[depthModel].to(device))
    # listDepth = listDepth + (depthDict[depthModel].to(device))
    # listDepth = tuple([depthDict[DepthConstSet().marigold].to(device),depthDict[DepthConstSet().midas].to(device),depthDict[DepthConstSet().depthAnythingV2].to(device)])
    return tuple(listDepth)

# command to start MLFlow = > mlflow server --host localhost --port 5000
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_classes: int = 13,
    epochs: int = 10,
    accumulation_steps: int = 2,
    learning_rate:float=1e-4,
    weight_decay:float=1e-3,
    device:torch.device=None,
    model_save_path:str="unet_nyu_depth_segmentation.pth",
    PD_ToTrain:str = DepthConstSet().raw,
    optimizer = None,
    use_scheduler = True,
    multi_step_list: list = [],
    RGB_ToTrain:bool = False,
    use_PDAM:bool = False,
    dim_PDAM:int = 1,
    pseudo_depth_models:tuple = [DepthConstSet().marigold,DepthConstSet().midas,DepthConstSet().depthAnythingV2],
    depth_all_zeros:bool = False,
    use_hha:bool = False,
    criterion = None,
    reduction_PDAM:int = 1
):
    """
    Trains the U-Net model for semantic segmentation on the NYU Depth dataset.

    Args:
        model (nn.Module): The U-Net model instance.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        num_classes (int): Number of output classes for segmentation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device, optional): Device to train on (e.g., 'cuda' or 'cpu').
                                         If None, it will be automatically detected.
        model_save_path (str): Path to save the best performing model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Move model to the specified device

    PDAM = PseudoDepthAggregationModule(dim=dim_PDAM, num_pseudo_depth_models=len(pseudo_depth_models),reduction=reduction_PDAM)
    PDAM.to(device)
    # # Set our tracking server uri for logging
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # # Create a new MLflow Experiment
    # expName = model_save_path.replace(".pth", "")
    # mlflow.set_experiment(expName)

    
    # Start an MLflow run
    # with mlflow.start_run():
    #     # Log the hyperparameters
    #     # mlflow.log_params(model.parameters())
    #     mlflow.log_param("learning_rate", learning_rate)
    #     mlflow.log_param("weight_decay", weight_decay)
    #     mlflow.log_param("epochs", epochs)
    #     mlflow.log_param("num_classes", num_classes)
    #     mlflow.log_param("batch_size", train_dataloader.batch_size)

        # Log the model
        # model_info = mlflow.pyfunc.log_model(name="model", python_model=model)
        # This saves the model and all its metadata for later retrieval.
        # mlflow.pytorch.log_model(model, "model", registered_model_name="expName")
        # mlflow.log_model_params(params={"param": "value"}, model_id=model_info.model_id)


    # # Instantiate optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # # For SGD with momentum, it would be:
    # # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
    # criterion = nn.BCELoss() 
    if criterion is None:
        criterion = nn.CrossEntropyLoss()    

    # --- CosineAnnealing + MultiStep ---
    if use_scheduler:
        if len(multi_step_list) == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * len(train_dataloader), eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=multi_step_list, gamma=0.1
            )


    # Initialize torchmetrics for Accuracy and JaccardIndex (mIoU)
    # The `task` parameter is crucial for correct metric calculation.
    # `num_classes` is important for multiclass metrics.
    train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    train_miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

    best_val_miou = -1.0 # Initialize with a low value to ensure first model is saved
    # patience = 3   # number of epochs to wait before triggering LR drop
    # no_improve_epochs = 0

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    train_mious = []
    val_losses = []
    val_accuracies = []
    val_mious = []

    first_batch = next(iter(train_dataloader))
    # image, (depth, label) = first_batch
    (img_test, _, _) = first_batch
    batch, _, height, width = img_test.shape
    # zero_depth_maps = torch.zeros((batch, 1, height, width), dtype=np.uint8).to(device)
    # zero_depth_map_hha = torch.zeros((batch, 3, height, width)).to(device)
    zero_depth_maps = torch.zeros((batch, 1, height, width), device=device)
    zero_depth_map_hha = torch.zeros((batch, 3, height, width), device=device)


    print(f"Training started on {device} for {epochs} epochs.")
    numSaveIdx = epochs*0.2
    numPrintBatch = int(len(train_dataloader)/10)

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Reset metrics for the current epoch
        train_accuracy_metric.reset()
        train_miou_metric.reset()
        optimizer.zero_grad() # Zero the gradients before each batch
        
        for batch_idx, (images, depthDict, GT) in enumerate(train_dataloader):
            rgb_images = images.to(device) # (N, 3, H, W)
            batch, _, height, width = rgb_images.shape

            if depth_all_zeros:
                depth_maps = torch.zeros((batch, 1, height, width), device=device)
            else:
                depth_maps = depthDict[PD_ToTrain].to(device) # (N, 1, H, W)
                if use_PDAM:
                    listDepth = get_listDepth(depthDict, device, pseudo_depth_models)
                    depth_maps = PDAM(listDepth)
            labels_one_hot = GT.to(device) # (N, num_classes, H, W) - FloatTensor (one-hot)

            # Concatenate RGB and Depth maps along the channel dimension
            if RGB_ToTrain:
                input_data = rgb_images
            else:
                input_data = torch.cat([rgb_images, depth_maps], dim=1) # Result: (N, 4, H, W)

            if use_hha:
                if depth_all_zeros:
                    depth_map_hha = torch.zeros((batch, 3, height, width), device=device)
                else:
                    depth_map_hha = depth_to_hha_Dataloader(depth_maps)
                loss_sub, loss_aux_sub, outputs = model(rgb_images, depth_map_hha, labels_one_hot)
                loss = loss_sub + loss_aux_sub * 0.2

            else:
            # optimizer.zero_grad() # Zero the gradients before each batch
                outputs = model(input_data) # Forward pass: (N, num_classes, H, W)

                loss = criterion(outputs, labels_one_hot) # Calculate loss
                
            loss.backward() # Backpropagate the loss
            if (batch_idx + 1) % accumulation_steps == 0:        
                optimizer.step()
                if use_scheduler:
                    scheduler.step()  # step per iteration, not per epoch
                optimizer.zero_grad()


            # optimizer.step() # Update model parameters        
            # scheduler.step()  # step per iteration, not per epoch

            running_loss += loss.item()
            # running_accuracy += calculate_accuracy(outputs, labels_one_hot)
            # running_miou += calculate_miou(outputs, labels_one_hot, num_classes)

            # For torchmetrics, convert one-hot labels back to class indices (LongTensor)
            # This is necessary because Accuracy and JaccardIndex with `task="multiclass"` expect target as class indices.
            labels_class_indices = torch.argmax(labels_one_hot, dim=1).long() # (N, H, W)
            # print(outputs.shape)
            # print(labels_class_indices.shape)

            # Update metrics
            train_accuracy_metric.update(outputs, labels_class_indices)
            train_miou_metric.update(outputs, labels_class_indices)

            # Print progress every 100 batches
            if (batch_idx + 1) % numPrintBatch == 0:
                current_train_acc = train_accuracy_metric.compute().item()
                current_train_miou = train_miou_metric.compute().item()        
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Train Loss: {running_loss / (batch_idx + 1):.4f}, "
                    f"Train Acc (current): {current_train_acc:.4f}, "
                    f"Train mIoU (current): {current_train_miou:.4f}" )


        # Calculate average training metrics for the epoch
        avg_train_loss = running_loss / len(train_dataloader)
        final_train_accuracy = train_accuracy_metric.compute().item()
        final_train_miou = train_miou_metric.compute().item()
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(final_train_accuracy)
        train_mious.append(final_train_miou)

        # # Log metrics with step tracking
        # mlflow.log_metrics({"train_loss": avg_train_loss, "train_accuracy": final_train_accuracy, "train_mIoU": final_train_miou}, step=epoch)


        print(f"Epoch {epoch+1} Training Summary - Loss: {avg_train_loss:.4f}, "
            f"Accuracy: {final_train_accuracy:.4f}, mIoU: {final_train_miou:.4f} ")
            # f"Current LR: {cosine_scheduler.get_last_lr()[0]:.6f}")   


        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
        val_loss = 0.0
        
        # Reset validation metrics for the current epoch
        val_accuracy_metric.reset()
        val_miou_metric.reset()

        with torch.no_grad(): # Disable gradient calculation during validation
            for (images, depthDict, GT) in val_dataloader:
                rgb_images = images.to(device)
                batch, _, height, width = rgb_images.shape
                
                if depth_all_zeros:
                    depth_maps = torch.zeros((batch, 1, height, width), device=device)
                else:
                    depth_maps = depthDict[PD_ToTrain].to(device) # (N, 1, H, W)
                
                if use_PDAM:
                    listDepth = get_listDepth(depthDict, device, pseudo_depth_models)
                    depth_maps = PDAM(listDepth)
                labels_one_hot = GT.to(device)

                if RGB_ToTrain:
                    input_data = rgb_images
                else:
                    input_data = torch.cat([rgb_images, depth_maps], dim=1)

                
                if use_hha:
                    if depth_all_zeros:
                        depth_map_hha = torch.zeros((batch, 3, height, width), device=device)
                    else:
                        depth_map_hha = depth_to_hha_Dataloader(depth_maps)
                    loss_sub, loss_aux_sub, outputs = model(rgb_images, depth_map_hha, labels_one_hot)
                    loss = loss_sub + loss_aux_sub * 0.2
                    
                else:
                    outputs = model(input_data)
                    loss = criterion(outputs, labels_one_hot)

                val_loss += loss.item()
                # val_accuracy += calculate_accuracy(outputs, labels_one_hot)
                # val_miou += calculate_miou(outputs, labels_one_hot, num_classes)

                labels_class_indices = torch.argmax(labels_one_hot, dim=1).long() # (N, H, W)

                # Update validation metrics
                val_accuracy_metric.update(outputs, labels_class_indices)
                val_miou_metric.update(outputs, labels_class_indices)

        # Calculate average validation metrics for the epoch
        avg_val_loss = val_loss / len(val_dataloader)
        final_val_accuracy = val_accuracy_metric.compute().item()
        final_val_miou = val_miou_metric.compute().item()
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(final_val_accuracy)
        val_mious.append(final_val_miou)

        # # Log validataion metrics with step tracking
        # mlflow.log_metrics({"val_loss": avg_val_loss, "val_accuracy": final_val_accuracy, "val_mIoU": final_val_miou}, step=epoch)

        print(f"Epoch {epoch+1} Validation Summary - Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {final_val_accuracy:.4f}, mIoU: {final_val_miou:.4f}")

        # Save the best model based on validation mIoU
        # if epoch > numSaveIdx:
        if  final_val_miou > best_val_miou:
            best_val_miou = final_val_miou
            # no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved validation mIoU: {best_val_miou:.4f}")
        else:
            # no_improve_epochs += 1
            print(f"Model not saved, current best validation mIoU: {best_val_miou:.4f}")
            
        # if no_improve_epochs >= patience:
        #     multistep_scheduler.step()
        #     no_improve_epochs = 0
        #     print(f"Validation mIoU plateaued. LR reduced to {multistep_scheduler.get_last_lr()[0]:.6f}")
 
        if use_scheduler:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")
        print("")

        # End run
        # mlflow.end_run()
    print("Training complete.")
    return train_losses, train_accuracies, train_mious, val_losses, val_accuracies, val_mious

def plot_metrics(train_losses, train_accuracies, train_mious, 
                 val_losses, val_accuracies, val_mious, epochs):
    """
    Plots training and validation metrics over epochs.

    Args:
        train_losses (list): List of training loss values per epoch.
        train_accuracies (list): List of training accuracy values per epoch.
        train_mious (list): List of training mIoU values per epoch.
        val_losses (list): List of validation loss values per epoch.
        val_accuracies (list): List of validation accuracy values per epoch.
        val_mious (list): List of validation mIoU values per epoch.
        epochs (int): Total number of epochs trained.
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(18, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o', markersize=4)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o', markersize=4)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot mIoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_mious, label='Training mIoU', marker='o', markersize=4)
    plt.plot(epochs_range, val_mious, label='Validation mIoU', marker='o', markersize=4)
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_model(
    model,
    test_dataloader,
    num_classes,
    device=None,
    model_load_path="unet_nyu_depth_segmentation.pth",
    visualize_samples=2, # Number of samples to visualize predictions for
    PD_ToTrain = DepthConstSet().raw,
    RGB_ToTrain = False,
    use_PDAM = False,
    dim_PDAM = 1,
    pseudo_depth_models:tuple = [DepthConstSet().marigold,DepthConstSet().midas,DepthConstSet().depthAnythingV2],
    depth_all_zeros:bool = False,
    use_hha:bool = False,
    reduction_PDAM = 1
):
    """
    Tests a trained U-Net model on a given DataLoader and visualizes predictions.

    Args:
        model (nn.Module): The U-Net model instance.
        test_dataloader (DataLoader): DataLoader for test/validation data.
        num_classes (int): Number of output classes for segmentation.
        device (torch.device, optional): Device to run on (e.g., 'cuda' or 'cpu').
                                         If None, it will be automatically detected.
        model_load_path (str): Path to the saved model state_dict.
        visualize_samples (int): Number of samples from the test_dataloader to visualize
                                 (original, ground truth, prediction).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    PDAM = PseudoDepthAggregationModule(dim=dim_PDAM, num_pseudo_depth_models=len(pseudo_depth_models), reduction=reduction_PDAM)
    PDAM.to(device)
    
    # Load the trained model state dictionary
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"Loaded model from {model_load_path}")
    else:
        print(f"Warning: Model not found at {model_load_path}. Using untrained model.")

    model.eval() # Set model to evaluation mode

    # Initialize torchmetrics for Accuracy and JaccardIndex (mIoU) for testing
    test_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

    print("\nStarting model testing and visualization...")
    visualized_count = 0

    with torch.no_grad(): # Disable gradient calculation
        for batch_idx, (images, depthDict, GT) in enumerate(test_dataloader):
            rgb_images = images.to(device)
            batch, _, height, width = rgb_images.shape
            
            if depth_all_zeros:
                depth_maps = torch.zeros((batch, 1, height, width), device=device)
            else:
                depth_maps = depthDict[PD_ToTrain].to(device)
                if use_PDAM:
                    listDepth = get_listDepth(depthDict, device, pseudo_depth_models)
                    depth_maps = PDAM(listDepth)
            
            labels_one_hot = GT.to(device)

            
            if RGB_ToTrain:
                input_data = rgb_images
            else:
                input_data = torch.cat([rgb_images, depth_maps], dim=1)

            
            if use_hha:
                if depth_all_zeros:
                    depth_map_hha = torch.zeros((batch, 3, height, width), device=device)
                else:
                    depth_map_hha = depth_to_hha_Dataloader(depth_maps)
                outputs = model(rgb_images, depth_map_hha)                
            
            else:
                outputs = model(input_data) # Model predictions (logits)

            # Convert one-hot labels to class indices for metric calculation
            labels_class_indices = torch.argmax(labels_one_hot, dim=1).long()
            # print(labels_class_indices.shape)

            # Update metrics
            test_accuracy_metric.update(outputs, labels_class_indices)
            test_miou_metric.update(outputs, labels_class_indices)

            # --- Visualize Predictions ---
            if visualized_count < visualize_samples:
                # Get predicted class map (N, H, W)
                predicted_class_map = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Get ground truth class map (N, H, W)
                ground_truth_class_map = labels_class_indices.cpu().numpy()

                # Denormalize RGB images for visualization
                # Need to get the original dataset instance from the DataLoader's dataset
                # This assumes test_dataloader.dataset is a Subset, and its .dataset is NyuDataset
                # original_nyu_dataset_instance = test_dataloader.dataset.dataset.dataset
                # # print(original_nyu_dataset_instance.normalize_rgb)
                # if original_nyu_dataset_instance.normalize_rgb:
                #     rgb_images_denorm = rgb_images.clone().cpu()
                #     for c in range(3):
                #         rgb_images_denorm[:, c, :, :] = rgb_images_denorm[:, c, :, :] * original_nyu_dataset_instance.std_rgb[c] + original_nyu_dataset_instance.mean_rgb[c]
                #     rgb_images_display = rgb_images_denorm.clamp(0, 1).permute(0, 2, 3, 1).numpy() # N, H, W, C
                # else:
                rgb_images_display = rgb_images.permute(0, 2, 3, 1).cpu().numpy() # N, H, W, C

                for i in range(rgb_images.shape[0]): # Iterate through samples in the current batch
                    if visualized_count >= visualize_samples:
                        break # Stop if we've visualized enough samples

                    plt.figure(figsize=(18, 6))
                    
                    # Original RGB Image
                    plt.subplot(1, 3, 1)
                    plt.imshow(rgb_images_display[i])
                    plt.title('Original RGB Image')
                    plt.axis('off')

                    # Ground Truth Segmentation
                    plt.subplot(1, 3, 2)
                    plt.imshow(ground_truth_class_map[i], cmap='tab20', vmin=0, vmax=num_classes - 1)
                    plt.title('Ground Truth Segmentation')
                    plt.colorbar(ticks=range(num_classes), fraction=0.046, pad=0.04)
                    plt.axis('off')

                    # Predicted Segmentation
                    print(predicted_class_map.shape)
                    plt.subplot(1, 3, 3)
                    plt.imshow(predicted_class_map[i], cmap='tab20', vmin=0, vmax=num_classes - 1)
                    plt.title('Predicted Segmentation')
                    plt.colorbar(ticks=range(num_classes), fraction=0.046, pad=0.04)
                    plt.axis('off')

                    plt.suptitle(f'Test Sample {batch_idx * test_dataloader.batch_size + i + 1}', fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()
                    visualized_count += 1
            
            if visualized_count >= visualize_samples:
                break # Stop processing batches if enough samples are visualized

    # Compute and print final test metrics
    final_test_accuracy = test_accuracy_metric.compute().item()
    final_test_miou = test_miou_metric.compute().item()
    
    # --- MLflow: Log final test metrics ---
    # mlflow.log_metric("test_accuracy", final_test_accuracy)
    # mlflow.log_metric("test_miou", final_test_miou)
    print(f"\n--- Test Results ---")
    print(f"Overall Test Accuracy: {final_test_accuracy:.4f}")
    print(f"Overall Test mIoU: {final_test_miou:.4f}")
    print("Testing complete.")


def BarChart_GFLOPS(results_latency_ms, results_gflops_per_sec, MODEL_NAME: str = "" ):
    if results_latency_ms:
        models_evaluated = list(results_latency_ms.keys())
        latencies = list(results_latency_ms.values())
        throughputs = list(results_gflops_per_sec.values())

        x = range(len(models_evaluated))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(14, 6))

        rects1 = ax1.bar([i - width/2 for i in x], latencies, width, label='Avg Latency (ms)', color='#d62728')
        ax1.set_ylabel('Latency (Lower is Better - ms)', color='#d62728', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#d62728')

        ax2 = ax1.twinx()
        rects2 = ax2.bar([i + width/2 for i in x], throughputs, width, label='Hardware GFLOPs/sec', color='#2ca02c')
        ax2.set_ylabel('Throughput (Higher is Better - GFLOPs/s)', color='#2ca02c', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')

        plt.title(f'Multi-Input Depth Model Benchmark Comparison {MODEL_NAME}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_evaluated, rotation=30, ha='right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        title = f'multi_input_comparison_benchmark_{MODEL_NAME}.png'
        plt.savefig(title, dpi=300)
        print(f"\n✅ Comparative chart saved as '{title}'")
        plt.show()