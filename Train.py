import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
from model_POSE_DCTNN_Single import DCTNN 
import random

def Freq_Butterworth_filter(x):
    a = torch.tensor([[0.010364       ,   0,  -0.010364]], dtype=torch.float32)
    b = torch.tensor([[1.0000 , -1.9781  , 0.9793]], dtype=torch.float32) 
    
    A = torch.fft.fft(a, n=2048)  
    B = torch.fft.fft(b, n=2048)  

    epsilon = 0#1e-6
    H = A/B
    H= (torch.abs(H).to(torch.float32))

    X = torch.fft.fft(x, n=2048)
    X = X * ((H)**2) 

    x = torch.fft.ifft(X, n=2048).real
    x= x[:, :512]
    
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Check for GPU

dr = "Data/Rodents7_Raw_Data"
percent = 1
x_train_mmap = np.load(os.path.join(dr,'x_train.npy'), mmap_mode='r')
x_train = x_train_mmap[:int(x_train_mmap.shape[0] * percent)]

# y_train
y_train_mmap = np.load(os.path.join(dr,'y_train.npy'), mmap_mode='r')
y_train = y_train_mmap[:int(y_train_mmap.shape[0] * percent)]

# x_val
x_val_mmap = np.load(os.path.join(dr,'x_val.npy'), mmap_mode='r')
x_val = x_val_mmap[:int(x_val_mmap.shape[0] * percent)]

# y_val
y_val_mmap = np.load(os.path.join(dr,'y_val.npy'), mmap_mode='r')
y_val = y_val_mmap[:int(y_val_mmap.shape[0] * percent)]



fraction = 1
num_samples_train = int(len(x_train) * fraction)
num_samples_test = int(len(x_val) * fraction)
# Slice the data
x_train = x_train[:num_samples_train]
y_train = y_train[:num_samples_train]



x_val = x_val[:num_samples_test]
y_val = y_val[:num_samples_test]





print(x_train.shape)
print(x_val.shape)



x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)

x_train = Freq_Butterworth_filter(x_train)
x_val = Freq_Butterworth_filter(x_val)

x_train = x_train / 6453
x_val = x_val / 6453

y_train_real = y_train[:, :1]
y_val_real = y_val[:, :1]
# Convert the numpy arrays to PyTorch tensors


y_train_imag = y_train[:, 1:]
y_val_imag = y_val[:, 1:]




# Combine real and imaginary targets
y_train_combined = torch.tensor(np.concatenate([y_train_real, y_train_imag], axis=1), dtype=torch.float32)
y_val_combined = torch.tensor(np.concatenate([y_val_real, y_val_imag], axis=1), dtype=torch.float32)

# Use a single dataset and loader
train_dataset = TensorDataset(x_train, y_train_combined)
test_dataset = TensorDataset(x_val, y_val_combined)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Initialize model
model = DCTNN().to(device)  # Assume model outputs 2 values per sample: real and imag
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Train loop
best_test_loss = float('inf')
best_epoch = 0
best_model_state = None
test_loss_log = []
test_mae_log = []
epochs = 7000

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute real and imaginary losses separately
        loss_real = criterion(outputs[:, 0], targets[:, 0])
        loss_imag = criterion(outputs[:, 1], targets[:, 1])
        loss = loss_real + loss_imag

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Eval phase
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_real = criterion(outputs[:, 0], targets[:, 0])
            loss_imag = criterion(outputs[:, 1], targets[:, 1])
            loss = loss_real + loss_imag
            test_loss += loss.item()

            mae_real = torch.nn.functional.l1_loss(outputs[:, 0], targets[:, 0])
            mae_imag = torch.nn.functional.l1_loss(outputs[:, 1], targets[:, 1])
            test_mae += (mae_real + mae_imag).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)

    test_loss_log.append(avg_test_loss)
    test_mae_log.append(avg_test_mae)

    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, Test MSE: {avg_test_loss:.6f}, Test MAE: {avg_test_mae:.6f}')

    # Save every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'New_Models_Freq_Butterworth2/final_model_Norm_Rodents7_Single_Random.pth')

    # Save best
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_epoch = epoch + 1
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, 'New_Models_Freq_Butterworth2/best_model_Norm_Rodents7_Single_Random.pth')

print(f'Best Test Loss: {best_test_loss:.6f} at Epoch {best_epoch}')