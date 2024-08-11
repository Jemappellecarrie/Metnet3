from metnet3_pytorch import MetNet3
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from dataloader import train_loader, val_loader  

# Initialized model 
metnet3 = MetNet3(
    dim=512,
    num_lead_times=722,
    lead_time_embed_dim=32,
    input_spatial_size=624,
    attn_dim_head=8,
    hrrr_channels=617,
    input_2496_channels=2 + 14 + 1 + 2 + 20,
    input_4996_channels=16 + 1,
    precipitation_target_bins=dict(
        mrms_rate=512,
        mrms_accumulation=512,
    ),
    surface_target_bins=dict(
        omo_temperature=256,
        omo_dew_point=256,
        omo_wind_speed=256,
        omo_wind_component_x=256,
        omo_wind_component_y=256,
    ),
    hrrr_loss_weight=10,
    hrrr_norm_strategy='sync_batchnorm',
    hrrr_norm_statistics=None
)

# CPU
device = torch.device("cpu")
metnet3.to(device)
optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# Save Path
save_path = "saved_models_2/"
os.makedirs(save_path, exist_ok=True)

train_losses = []
val_losses = []

# Train Loop
for epoch in range(100):  # 100 times
    metnet3.train()
    running_loss = 0.0

    for batch in train_loader:
        lead_times = batch['lead_times'].to(device)
        hrrr_input_2496 = batch['hrrr_input_2496'].to(device)
        hrrr_stale_state = batch['hrrr_stale_state'].to(device)
        input_2496 = batch['input_2496'].to(device)
        input_4996 = batch['input_4996'].to(device)
        precipitation_targets = {k: v.to(device) for k, v in batch['precipitation_targets'].items()}
        surface_targets = {k: v.to(device).long() for k, v in batch['surface_targets'].items()}
        hrrr_target = batch['hrrr_target'].to(device)

        optimizer.zero_grad()
        total_loss, loss_breakdown = metnet3(
            lead_times=lead_times,
            hrrr_input_2496=hrrr_input_2496,
            hrrr_stale_state=hrrr_stale_state,
            input_2496=input_2496,
            input_4996=input_4996,
            precipitation_targets=precipitation_targets,
            surface_targets=surface_targets,
            hrrr_target=hrrr_target
        )

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/100], Training Loss: {avg_loss}")  

    # Evaluate Model
    metnet3.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            lead_times = batch['lead_times'].to(device)
            hrrr_input_2496 = batch['hrrr_input_2496'].to(device)
            hrrr_stale_state = batch['hrrr_stale_state'].to(device)
            input_2496 = batch['input_2496'].to(device)
            input_4996 = batch['input_4996'].to(device)
            precipitation_targets = {k: v.to(device) for k, v in batch['precipitation_targets'].items()}
            surface_targets = {k: v.to(device).long() for k, v in batch['surface_targets'].items()}
            hrrr_target = batch['hrrr_target'].to(device)

            total_loss, loss_breakdown = metnet3(
                lead_times=lead_times,
                hrrr_input_2496=hrrr_input_2496,
                hrrr_stale_state=hrrr_stale_state,
                input_2496=input_2496,
                input_4996=input_4996,
                precipitation_targets=precipitation_targets,
                surface_targets=surface_targets,
                hrrr_target=hrrr_target
            )

            running_val_loss += total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/100], Validation Loss: {avg_val_loss}")

    # Save Model
    if (epoch + 1) % 10 == 0 or (epoch + 1) == 100:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': metnet3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }, os.path.join(save_path, f'metnet3_epoch_{epoch+1}.pth'))
        print(f"Model saved after epoch {epoch+1}")

# Visualization 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over 100 Epochs')
plt.legend()
plt.savefig(os.path.join(save_path, 'training_validation_loss.png'))
plt.show()

print("Training completed.")
