# Third party imports
import torch
import torch.nn.functional as F


def flow_matching_train_step(model, degraded_images, clean_images, optimizer):
    """
    model: Your U-Net (takes x_t and t, outputs predicted velocity)
    degraded_images: Tensor of shape (B, C, H, W) -> x_0
    clean_images: Tensor of shape (B, C, H, W) -> x_1
    """
    batch_size = degraded_images.shape[0]
    device = degraded_images.device

    # 1. Sample random time steps 't' for each image in the batch
    # t is drawn from a uniform distribution between 0 and 1
    t = torch.rand((batch_size,), device=device)
    
    # Reshape t so we can broadcast it across the image dimensions (B, 1, 1, 1)
    t_expanded = t.view(batch_size, 1, 1, 1)

    # 2. Calculate the intermediate image x_t (linear interpolation)
    x_t = (1.0 - t_expanded) * degraded_images + t_expanded * clean_images

    # 3. Calculate the target velocity (the straight line from degraded to clean)
    target_velocity = clean_images - degraded_images

    # 4. Forward pass: Predict the velocity
    # The model looks at the noisy blend (x_t) and the time (t)
    predicted_velocity = model(x_t, t)

    # 5. Calculate MSE Loss
    loss = F.mse_loss(predicted_velocity, target_velocity)

    # 6. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def euler_sample(model, degraded_image, num_steps=10):
    device = degraded_image.device
    x_t = degraded_image.clone()
    
    # Time step size
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        # Current time t
        t = torch.tensor([i * dt], device=device).expand(x_t.shape[0])
        
        # Predict velocity
        v_pred = model(x_t, t)
        
        # Take a step in the direction of the velocity
        x_t = x_t + v_pred * dt
        
    return x_t