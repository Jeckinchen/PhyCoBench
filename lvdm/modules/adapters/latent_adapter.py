import torch
import torch.nn as nn

class LatentAdapter(nn.Module):
    def __init__(self, frame_num, height, width, in_channels=4, out_channels=2):
        super(LatentAdapter, self).__init__()
        self.fc1 = nn.Linear(in_channels * frame_num * height * width, out_channels * frame_num * height * width)
        self.relu = nn.ReLU()  # 添加激活函数
        self.frame_num = frame_num
        self.height = height
        self.width = width

    def forward(self, x):
        #print("Forward method called")
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)  # 激活函数
        x = x.view(batch_size, -1, self.frame_num, self.height, self.width)  # 还原形状
        #print("Forward method completed")
        return x