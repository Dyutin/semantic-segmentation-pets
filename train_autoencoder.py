import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from autoencoder import AutoEncoder 
from pet_dataset import PetDataset
from torch.optim.lr_scheduler import StepLR
DATA_PATH = "./Dataset"
BATCH_SIZE = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = PetDataset(DATA_PATH)

total_samples = len(train_dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

autoencoder = AutoEncoder(in_channels=3).to(device)

optimizer_ae = torch.optim.AdamW(autoencoder.parameters(), lr=1e-4)
scheduler_ae = StepLR(optimizer_ae, step_size=10, gamma=0.5)
criterion_ae = nn.MSELoss()


num_epochs = 50
patience = 5  
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    autoencoder.train()
    running_loss = 0.0
    for img, _ in train_loader: 
        img = img.to(device)
        optimizer_ae.zero_grad()
        reconstructed = autoencoder(img)
        loss = criterion_ae(reconstructed, img)
        loss.backward()
        optimizer_ae.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    autoencoder.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for img, _ in val_loader:
            img = img.to(device)
            reconstructed = autoencoder(img)
            loss = criterion_ae(reconstructed, img)
            val_running_loss += loss.item()
    
    val_loss = val_running_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    scheduler_ae.step()
    print(f"Current learning rate: {optimizer_ae.param_groups[0]['lr']:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(autoencoder.state_dict(), "./Saved_Models/autoencoder_best.pth")
        print("Validation loss improved; saving model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")
