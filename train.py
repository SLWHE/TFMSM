
import os

from pygments.lexers import configs
from scipy.signal import freqs



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torch.optim as optim
import time
import scipy.io as sio
import math
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import gc
from normalize_code import normalize_matrix, denormalize_matrix
from cbd1dn import Network
from network import Network

input_dim = 1440
hidden_dim = 32
num_layers = 2
output_dim = 1

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = Network().to(device)


#model_path = 'cbc1dn_seabottom_lr00000_disorder-resmamba+100.pth'
#loaded_model = torch.load(model_path)
#net = loaded_model.to(device)


######
start = time.time()
EPOCH = 100
BATCH_SIZE_s = 80#batch_size
BATCH_SIZE_v = 80
LR = 0.000001  # 0.000001
rate = 0.9  #
iteration = 10

# training data
Origin_s = 'train_signal_choice.mat'
Origin_s = sio.loadmat(Origin_s)
Origin_s = Origin_s['a']
Origin_s = Origin_s[:, np.newaxis, :]


MTsignal_s = 'train_nosie_signal_choice.mat'
MTsignal_s = sio.loadmat(MTsignal_s)
MTsignal_s = MTsignal_s['b']
MTsignal_s = MTsignal_s[:, np.newaxis, :]


Ls = 79360
Origin_s = Origin_s[0:Ls, :, :]
MTsignal_s = MTsignal_s[0:Ls, :, :]


#validation data
Origin_v = 'val_signal_choice.mat'
Origin_v = sio.loadmat(Origin_v)
Origin_v = Origin_v['va']
Origin_v = Origin_v[:, np.newaxis, :]

MTsignal_v = 'val_nosie_signal_choice.mat'
MTsignal_v = sio.loadmat(MTsignal_v)
MTsignal_v = MTsignal_v['vb']
MTsignal_v = MTsignal_v[:, np.newaxis, :]
Lv = 19840
Origin_v = Origin_v[0:Lv, :, :]
MTsignal_v = MTsignal_v[0:Lv, :, :]


# Data Preprocessing
norm_s, row_min_s, row_max_s = normalize_matrix(Origin_s, MTsignal_s)
np.save('row_min_s', row_min_s)
np.save('row_max_s', row_max_s)
Origin_s = norm_s[0:Ls, :]
MTsignal_s = norm_s[Ls:Ls * 2, :]

norm_v, row_min_v, row_max_v = normalize_matrix(Origin_v, MTsignal_v)
np.save('row_min_v', row_min_v)
np.save('row_max_v', row_max_v)
Origin_v = norm_v[0:Lv, :]
MTsignal_v = norm_v[Lv:Lv * 2, :]

x1_s = torch.from_numpy(Origin_s)
x2_s = torch.from_numpy(MTsignal_s)
x1_s = x1_s.type(torch.FloatTensor)
x2_s = x2_s.type(torch.FloatTensor)

x1_v = torch.from_numpy(Origin_v)
x2_v = torch.from_numpy(MTsignal_v)
x1_v = x1_v.type(torch.FloatTensor)
x2_v = x2_v.type(torch.FloatTensor)

train_data = TensorDataset(x2_s, x1_s)
val_data = TensorDataset(x2_v, x1_v)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_s, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE_v, shuffle=False, num_workers=0, drop_last=True)


del Origin_s
del MTsignal_s
del Origin_v
del MTsignal_v
del norm_s
del norm_v
del x1_s
del x1_v
del x2_s
del x2_v
gc.collect()

###network setup###
criterion = nn.MSELoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.95)
StepLR = StepLR(optimizer, step_size=3, gamma=0.1)


#Start training
Losslist_s = []
Losslist_v = []
print("Start Training!")
for epoch in range(EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    if epoch % iteration == 9:
        LR = LR * rate
    loss_s = 0.0
    loss_v = 0.0
    for i, data_s in enumerate(train_loader, 0):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        input_s, target_s = data_s
        input_s = input_s.to(device)
        output_s = net(input_s)

        target_s = target_s.to(device)
        loss_s0 = criterion(output_s, target_s)
        loss_s0.backward()
        optimizer.step()
        loss_s += loss_s0.item()
    Losslist_s.append(loss_s / (Ls // BATCH_SIZE_s))
    net.eval()
    with torch.no_grad():
        for j, data_v in enumerate(val_loader, 0):
            input_v, target_v = data_v
            input_v = input_v.to(device)
            output_v = net(input_v)
            target_v = target_v.to(device)
            loss_v0 = criterion(output_v, target_v)
            loss_v += loss_v0.item()

        Losslist_v.append(loss_v / (Lv // BATCH_SIZE_v))
    print("第%d个epoch的学习率：%.10f" % (epoch + 1, optimizer.param_groups[0]['lr']))
    scheduler.step(loss_v)
    if (epoch + 1) % 1 == 0:
        print('train loss: {:.10f}'.format(loss_s / (Ls // BATCH_SIZE_s)))
        print('val loss: {:.10f}'.format(loss_v / (Lv // BATCH_SIZE_v)))
    results_file = "results.txt"
    with open(results_file, "a") as f:
        write_info = f"[epoch: {epoch}]  lr: {LR:.6f} train loss: {loss_s / (Ls // BATCH_SIZE_s):.10f} val loss: {loss_v / (Lv // BATCH_SIZE_v):.10f}\n"
        f.write(write_info)
    save_files = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch}
    if (epoch +1 ) % 10==0:
        torch.save(net, 'cbc1dn_seabottom_lr00000_disorder-Gnet1+{}.pth'.format(epoch + 1))
print('finished training')

###visualization###
input_v = input_v.cpu()
input_v = input_v.detach().numpy()
output_v = output_v.cpu()
output_v = output_v.detach().numpy()
target_v = target_v.cpu()
target_v = target_v.detach().numpy()

# Loss
x = range(1, EPOCH + 1)
y_s = Losslist_s
y_v = Losslist_v
plt.semilogy(x, y_s, 'b.-')
plt.semilogy(x, y_v, 'r.-')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.show()
plt.savefig("accuracy_loss.jpg")
torch.save(net, 'cbc1dn_seabottom_lr00000_disorder.pth')


###SNR###
origSignal = target_v
errorSignal = target_v - input_v
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues1 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues1)

#
origSignal = target_v
errorSignal = target_v - output_v
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues2 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues2)


for col in range(BATCH_SIZE_v):
    x = range(0, len(target_v[0, 0, :]))
    y1 = target_v[col, 0, :]
    y2 = input_v[col, 0, :]
    y3 = output_v[col, 0, :]
    plt.plot(x, y1, 'k.-')
    plt.plot(x, y2, 'r.-')
    plt.plot(x, y3, 'g.-')
    plt.xlabel('Time')
    plt.ylabel('Ampulitude')
    plt.show()



end = time.time()
print(end - start)
