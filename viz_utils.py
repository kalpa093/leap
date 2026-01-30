import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_random_direction(model, device):
    directions = {}
    for name, p in model.named_parameters():
        d = torch.randn(p.size(), device=device)
        # Filter Normalization
        if len(p.size()) > 1:
            d_norm = d.norm()
            p_norm = p.norm()
            if d_norm > 0 and p_norm > 0:
                d = d * (p_norm / (d_norm + 1e-10))
        directions[name] = d
    return directions

def prepare_batch(model, batch, device):
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    if model.config.is_encoder_decoder:
        if "labels" in inputs:
            if hasattr(model, "prepare_decoder_input_ids_from_labels"):
                inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(inputs["labels"])
            else:
                inputs["decoder_input_ids"] = torch.full(
                    (inputs["input_ids"].shape[0], 1),
                    model.config.pad_token_id, dtype=torch.long, device=device
                )
        else:
             inputs["decoder_input_ids"] = torch.full(
                (inputs["input_ids"].shape[0], 1),
                model.config.pad_token_id, dtype=torch.long, device=device
            )

    forward_keys = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
    model_inputs = {k: v for k, v in inputs.items() if k in forward_keys}
    return model_inputs

def save_loss_landscape(model, dataloader, device, save_path):
    NUM_BATCHES = 16 
    STEPS = 21       # (21x21 = 441 Grid points)
    RANGE_VAL = 0.5  # 
    
    print(f"Generating High-Fidelity Loss Landscape (Avg of {NUM_BATCHES} batches)...")
    model.eval()

    eval_batches = []
    try:
        for i, batch in enumerate(dataloader):
            if i >= NUM_BATCHES:
                break
            clean_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            eval_batches.append(clean_batch)
            
        print(f"Loaded {len(eval_batches)} batches for loss averaging.")
        if len(eval_batches) == 0:
            print("Warning: Dataloader is empty.")
            return

    except Exception as e:
        print(f"Error loading batches: {e}")
        return

    original_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    dir_x_dict = get_random_direction(model, device)
    dir_y_dict = get_random_direction(model, device)

    x_coords = np.linspace(-RANGE_VAL, RANGE_VAL, STEPS)
    y_coords = np.linspace(-RANGE_VAL, RANGE_VAL, STEPS)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros_like(X)

    total_grid_points = len(x_coords) * len(y_coords)
    print(f"Calculating Surface ({STEPS}x{STEPS} grid, Total {total_grid_points} points)...")

    step_count = 0
    
    with torch.no_grad():
        for i, alpha in enumerate(x_coords):
            for j, beta in enumerate(y_coords):
                
                for name, param in model.named_parameters():
                    if name in dir_x_dict and name in dir_y_dict:
                        d_x = dir_x_dict[name]
                        d_y = dir_y_dict[name]
                        orig_p = original_state_dict[name].to(device)
                        
                        param.data = orig_p + (alpha * d_x) + (beta * d_y)

                batch_losses = []
                for batch_idx, batch_data in enumerate(eval_batches):
                    model_inputs = prepare_batch(model, batch_data, device)
                    
                    outputs = model(**model_inputs)
                    
                    if outputs.loss is not None:
                        batch_losses.append(outputs.loss.item())
                
                if batch_losses:
                    avg_loss = sum(batch_losses) / len(batch_losses)
                else:
                    avg_loss = np.nan
                    
                Z[i, j] = avg_loss
                
                step_count += 1
                if step_count % 20 == 0:
                    print(f"Processing grid point {step_count}/{total_grid_points} | Current Loss: {avg_loss:.4f}")

    print("Restoring original model weights...")
    model.load_state_dict(original_state_dict)
    
    print("Plotting results...")
    try:
        if np.isnan(Z).any():
            Z = np.nan_to_num(Z, nan=np.nanmax(Z))

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
        plt.colorbar(cp, label='Loss Value')
        
        center_idx = STEPS // 2
        center_loss = Z[center_idx, center_idx]
        plt.scatter(0, 0, c='red', marker='*', s=150, label='Current Model')
        
        plt.title(f'Loss Landscape (Avg of {len(eval_batches)} Batches)\nCenter Loss: {center_loss:.4f}')
        plt.xlabel('Direction X')
        plt.ylabel('Direction Y')
        plt.legend()
        
        save_path_2d = save_path
        plt.savefig(save_path_2d, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
        
        ax.set_title(f'Loss Landscape 3D (Avg of {len(eval_batches)} Batches)', fontsize=15)
        ax.set_xlabel('Direction X')
        ax.set_ylabel('Direction Y')
        ax.set_zlabel('Loss')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss Value')
        ax.view_init(elev=30, azim=45)
        
        save_path_3d = save_path.replace(".pdf", "_3d.pdf")
        plt.savefig(save_path_3d, bbox_inches='tight')
        plt.close()

        print(f" High-Fidelity landscapes saved:\n   - {save_path_2d}\n   - {save_path_3d}")

    except Exception as e:
        print(f" Error plotting landscape: {e}")
