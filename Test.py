import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
#from MLP import MLFilter 
import random
from scipy.signal import butter, sosfiltfilt
from model_POSE_DCTNN_Single import DCTNN  

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
# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

#Remove this if you want to plot on the screen
matplotlib.use('Agg')

# Load test data from .npy files
# x_test = np.load('Data/x_test_raw_nz04_6111595_20221222_8_theta.npy')
# y_test = np.load('Data/y_test_filtered_nz04_6111595_20221222_8_theta.npy')


x_test = np.load('Data/Rodents8_Raw_Data/x_test.npy')
y_test = np.load('Data/Rodents8_Raw_Data/y_test.npy')

print(x_test.shape)
print(y_test.shape)

y_test_real = y_test[:, :1]
y_test_imag = y_test[:, 1:]

#y_test_tensor = torch.tensor(np.concatenate([y_test_real, y_test_imag], axis=1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
print(y_test_tensor.shape)
# Convert downsampled x_test and y_test to PyTorch tensors
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# Apply the filter to the test data
x_test_tensor = Freq_Butterworth_filter(x_test_tensor)
x_test_tensor = x_test_tensor / 6453

# Create TensorDataset and DataLoader for the test dataset
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Check for GPU

# Initialize models and load the saved states
model = DCTNN().to(device)
#print(model)
model.load_state_dict(torch.load('New_Models_Freq_Butterworth2/best_model_Norm_Rodents7_Single_Random.pth',weights_only=False))


#Evaluate
model.eval()
print(sum(p.numel() for p in model.parameters()))
print("Weights of filter1 :")
print(model.DownsampleLayer1.filter1.weight.data)

print("Weights of filter2 :")
print(model.DownsampleLayer1.filter2.weight.data)
print("Weights of filter1 :")
print(model.DownsampleLayer2.filter1.weight.data)

print("Weights of filter2 :")
print(model.DownsampleLayer2.filter2.weight.data)
#predictions = []

real_predictions = []
imag_predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        
        outputs = model(inputs.to(device))
        
        #predictions.append(outputs.cpu().numpy()) 
        real_predictions.append(outputs[:, 0].cpu().numpy()) 
        imag_predictions.append(outputs[:, 1].cpu().numpy())

real_predictions = np.concatenate(real_predictions, axis=0).reshape(-1, 1)
imag_predictions = np.concatenate(imag_predictions, axis=0).reshape(-1, 1)

phase_pred = np.arctan2(imag_predictions, real_predictions)
phase_real = np.arctan2(y_test_imag, y_test_real)
print('Phase Prediction:',phase_pred.shape)
print('Phase Real:',phase_real.shape)

start_idx = 30000
end_idx = 32000


#Calculate angels and Plot predictions vs true values
Phase_pred = phase_pred[start_idx:end_idx].reshape(-1)
Phase_true = phase_real[start_idx:end_idx].reshape(-1)

real_pred_part = real_predictions[start_idx:end_idx].reshape(-1)
real_true_part = y_test_real[start_idx:end_idx].reshape(-1)

imag_pred_part = imag_predictions[start_idx:end_idx].reshape(-1)
imag_true_part = y_test_imag[start_idx:end_idx].reshape(-1)

sampling_rate = 1500  
time_axis = np.arange(0, len(Phase_pred)) / sampling_rate

plt.figure(figsize=(12, 6))
plt.plot(time_axis, Phase_true, color='gray', label='True Phase')
plt.plot(time_axis, Phase_pred, color='blue', label='Predicted Phase')
plt.legend()
plt.title('True Phase vs Predicted Phase')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase')
plt.savefig('Phase.png')
plt.show()


# Plotting Real Part Comparison
plt.figure(figsize=(12, 6))
plt.plot(time_axis, real_true_part, color='gray', label='True Real Part')
plt.plot(time_axis, real_pred_part, color='blue', label='Predicted Real Part')
plt.legend()
plt.title('True Real Part vs Predicted Real Part')
plt.xlabel('Time (seconds)')
plt.ylabel('Real Part')
plt.savefig('Real Part.png')
plt.show()

# Plotting Imaginary Part Comparison
plt.figure(figsize=(12, 6))
plt.plot(time_axis, imag_true_part, color='gray', label='True Imaginary Part')
plt.plot(time_axis, imag_pred_part, color='blue', label='Predicted Imaginary Part')
plt.legend()
plt.title('True Imaginary Part vs Predicted Imaginary Part')
plt.xlabel('Time (seconds)')
plt.ylabel('Imaginary Part')
plt.savefig('Imaginary Part.png')
plt.show()

# Evaluation metrics

def RMS_estimation_error( estimated_phases,ground_truth_phases):
    """
    Calculate the RMS of circular phase differences between ground truth and estimated phases.

    Parameters:
    ground_truth_phases (numpy array): Array of ground truth phases (in radians).
    estimated_phases (numpy array): Array of estimated phases (in radians).

    Returns:
    float: RMS of the circular phase differences.
    """    
    # Calculate circular error
    estm_error = np.abs(estimated_phases - ground_truth_phases)
    circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)

    # Calculate RMS of the circular differences
    rms = np.sqrt(np.mean(circular_estm_error**2))

    return rms


def Mean_estimation_error(
    estimated_phases,
    ground_truth_phases
):
    '''
    Calculate mean phase estimation error

    Parameters
    -------------------------------
    estimated_phases: estimated phases by the real time algorithm
    ground_truth_phases: asssumed to have the SAME LENGTH as the estimated phases
    '''
    estm_error = np.abs(estimated_phases - ground_truth_phases)
    circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)
    return np.mean(circular_estm_error)

def circ_diff(unit1, unit2):
    """
    Calculate smallest difference between two angles for arrays or scalars.
    :param unit1: angle in radians (array or scalar)
    :param unit2: angle in radians (array or scalar)
    :return: difference (array or scalar)
    """
    # 1) mod 2π
    phi = np.remainder(unit2 - unit1, 2 * np.pi)
    # 2)  phi > π
    result = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    return result

#RMSE and MAE 
original_signals = phase_real.reshape(-1)
predicted_signals = phase_pred.reshape(-1)

RMS = RMS_estimation_error(original_signals, predicted_signals)
Mean = Mean_estimation_error(original_signals, predicted_signals)
print("RMSE:",RMS)
print("MAE:",Mean)

# Quantitative evaluation
phase_pred = np.arctan2(imag_predictions, real_predictions).reshape(-1)
phase_real = np.arctan2(y_test_imag, y_test_real).reshape(-1)

plt.figure()
plt.scatter(phase_real, phase_pred, s=0.001)
plt.xlabel('Ground truth phase')
plt.ylabel('Predicted phase')

diagonal = np.arange(-np.pi, np.pi)
plt.plot(diagonal, diagonal, ls='--', c='r')
plt.savefig('Quantization.png')
plt.axis('square')

# Circular Estimation Error

circular_err = circ_diff(phase_pred, phase_real)
circular_err = circular_err.reshape(-1)

num_bins = 10 
bins = np.linspace(-np.pi, np.pi, num_bins + 1)

# Bin the data based on ground truth phases
bin_indices = np.digitize(phase_real, bins)
binned_errors = [circular_err[bin_indices == i] for i in range(1, len(bins))]

plt.figure()
plt.boxplot(
    binned_errors, 
    positions=(bins[:-1] + bins[1:]) / 2, 
    widths=np.pi / num_bins,
    showfliers=False,
    medianprops={'linewidth': 2.5}
)

plt.axhline(y=0, ls='--', c='k', alpha=0.3)

plt.xlabel("Ground truth phase")
plt.ylabel("Circular error")
plt.title("Circular estimation errors\n binned by ground truth phase")
plt.xticks(
    ticks=(bins[:-1] + bins[1:]) / 2, 
    labels=[f"{round(x, 2)}" for x in (bins[:-1] + bins[1:]) / 2],
    rotation=25
)
plt.tight_layout()
plt.ylim(-np.pi, np.pi)
plt.savefig('Circular estimation errors binned by ground truth phase.png')
plt.axis('square')

