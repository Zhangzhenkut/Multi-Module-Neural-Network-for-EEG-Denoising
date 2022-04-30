import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self,c, k, T):
        super(block, self).__init__() 
        self.covn1d_1 = nn.Conv1d(in_channels = 1, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_4 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.fc_1 = nn.Linear(c * T, T)
        self.fc_2 = nn.Linear(c * T, T)
        
    def forward(self, x):
        x = torch.relu(self.covn1d_1(x))
        x = torch.relu(self.covn1d_2(x) + x)    
        x = torch.relu(self.covn1d_3(x) + x)
        x = torch.relu(self.covn1d_4(x) + x)
        signal =  self.fc_1(x.view(x.size(0),-1)).unsqueeze(1)
        noise = self.fc_2(x.view(x.size(0),-1)).unsqueeze(1)
        return signal, noise
class MMNN_4(nn.Module):
    def __init__(self,c, k, T):
        super(MMNN_4, self).__init__()
        self.block_1 = block(c, k, T)
        self.block_2 = block(c, k, T)
        self.block_3 = block(c, k, T)
        self.block_4 = block(c, k, T)
    def forward(self, x):
        signal_1, noise_1 = self.block_1(x)
        signal_2, noise_2 = self.block_2(x - noise_1)
        signal_3, noise_3 = self.block_3(x - noise_2)
        signal_4, noise_4 = self.block_4(x - noise_3)
        return signal_1 + signal_2 + signal_3 + signal_4

# For details of nn.Conv1d: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
x = torch.ones(1,1,512)
net = MMNN_4(32,25,512)
print('MMNN-4 for OA removal',net(x).size())
x = torch.ones(1,1,1024)
net = MMNN_4(32,103,1024)
print('MMNN-4 for MA removal',net(x).size())