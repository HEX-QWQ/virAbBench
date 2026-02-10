import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        """
        堆叠的 CNN 模块，用于特征提取和空间降维。
        
        Args:
            in_channels: 输入通道数 (hidden_size)。
            out_channels_list: 每层 CNN 的输出通道数列表。
        """
        super(CNNBlock, self).__init__()
        layers = []
        curr_channels = in_channels
        for out_channels in out_channels_list:
            layers.append(nn.Conv1d(curr_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            curr_channels = out_channels
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, hidden_size)
        Returns:
            Flattened features: shape (batch, out_channels * reduced_seq_len)
        """
        # Conv1d 期望输入为 (batch, channels, length)
        # 所以需要转置：(batch, seq_len, hidden_size) -> (batch, hidden_size, seq_len)
        x = x.transpose(1, 2)
        x = self.net(x)
        # 拉平
        return x.flatten(1)

class AffinityPredictor(nn.Module):
    def __init__(self, hidden_size=1536, seq_len=128):
        """
        亲和力预测模型。
        
        Args:
            hidden_size: Backbone 输出的隐藏层维度。
            seq_len: 输入序列的固定长度。
        """
        super(AffinityPredictor, self).__init__()
        
        # 设计 CNN 降维路径：1536 -> 512 -> 256 -> 128 -> 64
        # 对应的 seq_len 变化：128 -> 64 -> 32 -> 16 -> 8
        cnn_channels = [512, 256, 128, 64]
        self.cnn = CNNBlock(hidden_size, cnn_channels)
        
        # 计算 CNN 降维后的总特征维度
        # 每经过一层 MaxPool1d(2)，seq_len 减半
        reduced_seq_len = seq_len // (2 ** len(cnn_channels))
        final_cnn_out_dim = cnn_channels[-1] * reduced_seq_len
        
        # MLP 结构
        # 输入是 heavy, light, antigen 三者特征拼接
        self.mlp = nn.Sequential(
            nn.Linear(final_cnn_out_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 二分类：0/1 label
        )
        
    def forward(self, heavy, light, antigen):
        """
        Args:
            heavy: (batch, seq_len, hidden_size)
            light: (batch, seq_len, hidden_size)
            antigen: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, 2)
        """
        # 分别提取特征并拉平
        h_feat = self.cnn(heavy)
        l_feat = self.cnn(light)
        a_feat = self.cnn(antigen)
        
        # 拼接到一起
        combined = torch.cat([h_feat, l_feat, a_feat], dim=1)
        
        # 输入 MLP
        logits = self.mlp(combined)
        return logits

if __name__ == "__main__":
    # 模拟输入数据进行测试
    batch_size = 4
    seq_len = 128
    hidden_size = 1536
    
    model = AffinityPredictor(hidden_size=hidden_size, seq_len=seq_len)
    
    h = torch.randn(batch_size, seq_len, hidden_size)
    l = torch.randn(batch_size, seq_len, hidden_size)
    a = torch.randn(batch_size, seq_len, hidden_size)
    
    output = model(h, l, a)
    print(f"Input shape: {h.shape}")
    print(f"Output shape: {output.shape}") # 应该是 (4, 2)
    print("Forward pass successful!")
