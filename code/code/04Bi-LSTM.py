import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib

class InjuryRiskDataset(Dataset):
    """
    运动科学损伤风险数据集
    将NumPy数组封装成PyTorch Dataset对象
    """
    def __init__(self, X_data, y_data):
        # 转换为torch.Tensor，确保数据类型正确
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)  # 调整为(n_samples, 1)
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

class BiLSTMFeatureExtractor(nn.Module):
    """
    Bi-LSTM特征提取器
    
    这是一个专门设计用于运动科学时序分析的Bi-LSTM模型。
    它的目标不是直接分类，而是提取有意义的32维特征向量，
    这些特征将用于后续的XGBoost集成模型。
    
    研究基础: 双向LSTM能够同时捕获时序数据的前向和后向依赖关系，
    这对于理解运动员的疲劳累积和损伤风险模式至关重要。
    """
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, output_size=32):
        """
        初始化Bi-LSTM特征提取器
        
        参数:
            input_size: 输入特征维度 (22个运动科学指标)
            hidden_size: LSTM隐藏层维度 (64维)
            num_layers: LSTM层数 (2层，增强特征提取能力)
            output_size: 最终提取的特征维度 (32维，符合研究摘要要求)
        """
        super(BiLSTMFeatureExtractor, self).__init__()
        
        # 模型参数 - 存储用于后续参考
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Bi-LSTM层 - 核心架构
        # bidirectional=True: 启用双向处理，捕获前后时序依赖
        # batch_first=True: 匹配我们的数据格式 [batch, time, features]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1  # 添加dropout防止过拟合
        )
        
        # 激活函数 - ReLU提供非线性变换能力
        self.relu = nn.ReLU()
        
        # 输出线性层 - 将Bi-LSTM输出映射到目标特征维度
        # hidden_size * 2: 因为是双向LSTM，输出维度翻倍
        self.linear = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        """
        前向传播过程
        
        参数:
            x: 输入张量，形状 [batch_size, 32, 22]
            
        返回:
            features: 提取的32维特征向量，形状 [batch_size, 32]
        """
        # LSTM前向传播
        # output形状: [batch_size, 32, hidden_size * 2] (双向输出)
        # h_n形状: [num_layers * 2, batch_size, hidden_size] (最终隐藏状态)
        # c_n形状: [num_layers * 2, batch_size, hidden_size] (最终细胞状态)
        output, (h_n, c_n) = self.lstm(x)
        
        # 关键：时序池化 - 取最后一个时间步的输出
        # 这代表了整个32天时序窗口的累积信息
        # output[:, -1, :] 形状: [batch_size, hidden_size * 2]
        pooled_output = output[:, -1, :]
        
        # 通过ReLU激活函数 - 引入非线性
        activated = self.relu(pooled_output)
        
        # 通过线性层得到最终32维特征向量
        # features形状: [batch_size, 32]
        features = self.linear(activated)
        
        return features

def calculate_class_weights(y_data):
    """
    计算类别权重以处理极端类别不平衡
    
    在运动科学中，损伤事件极其罕见（约0.5%），
    但预测这些罕见事件是我们的核心目标。
    
    参数:
        y_data: 标签数组
        
    返回:
        pos_weight: 正样本权重系数
    """
    unique, counts = np.unique(y_data, return_counts=True)
    print(f'类别分布: {{0: {counts[0]}, 1: {counts[1]}}}')
    
    # 计算pos_weight: 负样本数量 / 正样本数量
    pos_weight = counts[0] / counts[1]
    print(f'正样本权重 (pos_weight): {pos_weight:.2f}')
    print(f'这意味着模型将{pos_weight:.0f}倍地关注正样本（损伤）')
    
    return pos_weight

def main():
    """
    主函数 - 整合所有组件
    """
    print('=== PyTorch Bi-LSTM特征提取器构建开始 ===')
    print(f'PyTorch版本: {torch.__version__}')
    print()

    # =============================================================================
    # 加载数据
    # =============================================================================
    print('加载X_samples.npy和y_labels.npy数据...')
    X_data = np.load(r'E:\AIMeeting\X_samples.npy')
    y_data = np.load(r'E:\AIMeeting\y_labels.npy')

    print(f'X_data形状: {X_data.shape}')
    print(f'y_data形状: {y_data.shape}')
    print()

    # =============================================================================
    # 创建数据集
    # =============================================================================
    print('=== 第一步：创建PyTorch数据集 ===')
    print()
    
    print('实例化InjuryRiskDataset...')
    dataset = InjuryRiskDataset(X_data, y_data)
    print(f'Dataset创建完成，总样本数: {len(dataset)}')

    # 测试Dataset的功能
    sample_X, sample_y = dataset[0]
    print(f'样本X形状: {sample_X.shape}')
    print(f'样本y形状: {sample_y.shape}')
    print()

    # =============================================================================
    # 定义模型架构
    # =============================================================================
    print('=== 第二步：定义Bi-LSTM模型架构 ===')
    print()
    
    print('模型设计原理:')
    print('- 双向LSTM：同时捕获时序数据的前向和后向依赖关系')
    print('- 特征提取：输出32维特征向量供XGBoost使用')
    print('- 时序池化：取最后一个时间步的累积信息')
    print()

    # =============================================================================
    # 实现类别加权损失
    # =============================================================================
    print('=== 第三步：实现类别加权损失 ===')
    print()

    # 计算类别权重
    print('计算类别权重以处理极端类别不平衡...')
    pos_weight = calculate_class_weights(y_data)

    # 转换为torch.Tensor
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
    print()

    # 定义损失函数
    print('定义加权BCE损失函数...')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print('损失函数创建完成！')
    print()

    # =============================================================================
    # 整合与报告
    # =============================================================================
    print('=== 第四步：整合与报告 ===')
    print()

    # 实例化模型
    print('实例化BiLSTMFeatureExtractor模型...')
    model = BiLSTMFeatureExtractor(
        input_size=22,      # 特征数量
        hidden_size=64,     # LSTM隐藏层维度
        num_layers=2,       # LSTM层数
        output_size=32      # 最终提取的特征维度
    )

    print('模型结构:')
    print(model)
    print()

    # 测试模型
    print('测试模型前向传播...')
    batch_size = 32
    test_input = torch.randn(batch_size, 32, 22)  # [batch_size, time_steps, features]
    with torch.no_grad():
        output_features = model(test_input)

    print(f'输入形状: {test_input.shape}')
    print(f'输出特征形状: {output_features.shape}')
    print(f'期望输出形状: ({batch_size}, 32)')
    print(f'形状验证: {output_features.shape == (batch_size, 32)}')
    print()

    # 模型参数统计
    print('模型参数统计:')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数数量: {total_params:,}')
    print(f'可训练参数数量: {trainable_params:,}')
    print()

    # 保存模型组件供后续使用
    print('保存模型组件供后续使用...')
    torch.save(model.state_dict(), r'E:\AIMeeting\bilstm_model.pth')
    joblib.dump(criterion, r'E:\AIMeeting\loss_function.pkl')
    
    print('模型组件已保存:')
    print('- bilstm_model.pth (Bi-LSTM模型权重)')
    print('- loss_function.pkl (加权损失函数)')
    print()

    print('=== Bi-LSTM特征提取器构建完成！===')
    print()
    print('Target 组件摘要:')
    print(f'1. Dataset: InjuryRiskDataset ({len(dataset)} 样本)')
    print(f'2. Model: BiLSTMFeatureExtractor (输入: 22维, 输出: 32维特征)')  
    print(f'3. Loss: BCEWithLogitsLoss (pos_weight: {pos_weight:.2f})')
    print()
    print('[OK] 模型已准备好进行特征提取训练！')
    print('[OK] 32维特征输出将用于XGBoost集成！')
    
    return dataset, model, criterion

if __name__ == "__main__":
    dataset, model, criterion = main()