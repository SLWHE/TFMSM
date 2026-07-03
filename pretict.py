import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import savemat
#from network import Network as net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load model
model_path = 'cbc1dn_seabottom_lr00000_disorder.pth'
net = torch.load(model_path).to(device)
#net = net().to(device)
#epoch_to_load = 40
#net.load_state_dict(torch.load(f'generator_epoch_{epoch_to_load}.pth'))
net.eval()


def normalize_matrix(matrix):

    row_min = np.min(matrix, axis=2, keepdims=True)
    midvar = matrix - row_min
    row_max = np.max(np.abs(midvar), axis=2, keepdims=True)
    row_max_true = row_max
    row_max[row_max == 0] = 1
    normalized_matrix = midvar  / row_max

    return normalized_matrix, row_min, row_max_true


def denormalize_matrix(normalized_matrix, row_min, row_max):
    denormalized_matrix = normalized_matrix * row_max + row_min
    return denormalized_matrix

real_noisy_data_path = 'test-noise.mat'  # Please replace with the actual file path of real noisy data.
real_noisy_data = sio.loadmat(real_noisy_data_path)
real_noisy_data = real_noisy_data['b']
real_noisy_data = real_noisy_data[:, np.newaxis, :]
norm_real_noisy_data, row_min, row_max = normalize_matrix(real_noisy_data)

real_noisy_signal = torch.from_numpy(norm_real_noisy_data).type(torch.FloatTensor)

batch_size = 8
real_noisy_data = TensorDataset(real_noisy_signal)
real_noisy_loader = DataLoader(real_noisy_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

# predict test samples
output_list = []
x_list = []
cA1_output_list = []
cD1_output_list =[]
with torch.no_grad():
    for i, data in enumerate(real_noisy_loader, 0):
        input_data = data[0].to(device)
        output = net(input_data)
        output_list.append(output.cpu().detach().numpy())

output = np.concatenate(output_list, axis=0)

output = denormalize_matrix(output, row_min, row_max)

for col in range(min(64,len(output))):
    x = range(0, len(real_noisy_signal[0, 0, :]))
    y1 = real_noisy_signal[col, 0, :]
    y2 = output[col, 0, :]
    plt.plot(x, y1, 'r.-', label='Noisy Signal')
    plt.plot(x, y2, 'g.-', label='Denoised Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()



def save_to_mat(data, output_file):
    try:
        savemat(output_file, {'data': data})
        print(f"The data has been successfully saved to {output_file}")
    except Exception as e:
        print(f"Error while saving data to {output_file}: {e}")

output_file = 'test-denoise.mat'
denosie = np.squeeze(output)
save_to_mat(denosie, output_file)
