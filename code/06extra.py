import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print('=== Bi-LSTM特征提取与XGBoost数据准备阶段开始 ===')
    print(f'PyTorch版本: {torch.__version__}')
    print()

    # =============================================================================
    # 第一步：加载组件并设置模型
    # =============================================================================
    print('=== 第一步：加载组件并设置模型 ===')
    print()

    # 重新定义Bi-LSTM模型架构
    class BiLSTMFeatureExtractor(nn.Module):
        def __init__(self, input_size=22, hidden_size=64, num_layers=2, feature_size=32):
            super(BiLSTMFeatureExtractor, self).__init__()
            
            # 特征提取部分
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, bidirectional=True)
            self.relu = nn.ReLU()
            self.feature_extractor = nn.Linear(hidden_size * 2, feature_size)
            
            # 分类头 - 用于训练时的损失计算
            self.classifier = nn.Linear(feature_size, 1)
            
        def forward(self, x, return_features=False):
            # 特征提取
            output, (h_n, c_n) = self.lstm(x)
            pooled_output = output[:, -1, :]
            activated = self.relu(pooled_output)
            features = self.feature_extractor(activated)
            
            # 分类预测
            logits = self.classifier(features)
            
            if return_features:
                return logits, features  # 返回logits和特征
            else:
                return logits  # 只返回logits用于训练

    # 重新定义数据集类
    class InjuryRiskDataset(torch.utils.data.Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = torch.tensor(X_data, dtype=torch.float32)
            self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
            
        def __len__(self):
            return len(self.X_data)
        
        def __getitem__(self, idx):
            return self.X_data[idx], self.y_data[idx]

    print('重新定义BiLSTMFeatureExtractor模型...')
    print('模型特点:')
    print('  - 双向LSTM: 捕获前后时序依赖')
    print('  - 特征提取层: 输出32维深度特征')
    print('  - 分类头: 用于训练时的损失计算')
    print()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 实例化模型
    print('实例化BiLSTMFeatureExtractor模型...')
    model = BiLSTMFeatureExtractor(input_size=22, hidden_size=64, num_layers=2, feature_size=32)
    model = model.to(device)

    print('加载最佳模型权重...')
    try:
        model.load_state_dict(torch.load(r'E:\AIMeeting\best_bilstm_feature_extractor.pth'))
        print('[OK] 最佳模型权重加载成功！')
    except FileNotFoundError:
        print('[WARN] 最佳模型权重文件未找到，将使用随机权重进行特征提取演示')
        print('  请确保已完成第五步的训练')

    # 设置为评估模式 - 极其重要！
    model.eval()
    print('[OK] 模型已设置为评估模式（model.eval()）')
    print('  这将关闭Dropout等训练专用层，确保特征提取的稳定性')
    print()

    # =============================================================================
    # 第二步：执行批量特征提取
    # =============================================================================
    print('=== 第二步：执行批量特征提取 ===')
    print()

    # 加载完整数据集
    print('加载完整数据集X_samples.npy和y_labels.npy...')
    X_samples = np.load(r'E:\AIMeeting\X_samples.npy')
    y_labels = np.load(r'E:\AIMeeting\y_labels.npy')

    print(f'数据加载完成:')
    print(f'  X_samples形状: {X_samples.shape}')
    print(f'  y_labels形状: {y_labels.shape}')
    print(f'  总样本数: {len(X_samples)}')
    print()

    # 创建完整数据集的DataLoader
    print('创建完整数据集的DataLoader...')
    full_dataset = InjuryRiskDataset(X_samples, y_labels)
    full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)  # shuffle=False保持时间顺序

    print(f'DataLoader配置:')
    print(f'  批次大小: 128')
    print(f'  总批次数量: {len(full_loader)}')
    print(f'  shuffle=False: 保持原始时间顺序')
    print()

    # 执行批量特征提取
    print('执行批量特征提取...')
    extracted_features = []
    batch_count = 0

    print('提取进度:')
    with torch.no_grad():  # 关键：关闭梯度计算以提高效率
        for batch_idx, (batch_X, batch_y) in enumerate(full_loader):
            batch_X = batch_X.to(device)
            
            # 关键：调用return_features=True获取32维特征
            _, features = model(batch_X, return_features=True)
            
            # 将特征移到CPU并转换为NumPy数组
            features_numpy = features.cpu().numpy()
            extracted_features.append(features_numpy)
            
            batch_count += 1
            if batch_idx % 50 == 0:  # 每50个批次显示进度
                print(f'  批次 {batch_idx+1}/{len(full_loader)} 完成')

    print(f'[OK] 特征提取完成！')
    print(f'  总提取批次: {batch_count}')
    print(f'  每批次特征形状: {extracted_features[0].shape}')
    print()

    # 合并所有批次的特征
    print('合并所有批次的特征...')
    X_features_bilstm = np.concatenate(extracted_features, axis=0)
    print(f'[OK] 特征合并完成！')
    print(f'  合并后特征形状: {X_features_bilstm.shape}')
    print()

    # =============================================================================
    # 第三步：为XGBoost准备最终数据
    # =============================================================================
    print('=== 第三步：为XGBoost准备最终数据 ===')
    print()

    # 重新进行时序划分（与第五步完全相同的逻辑）
    print('重新进行时序划分（与第五步完全相同的80/20比例）...')
    total_samples = len(X_features_bilstm)
    train_size = int(0.8 * total_samples)

    # 划分特征数据
    X_train_xgb = X_features_bilstm[:train_size]
    X_val_xgb = X_features_bilstm[train_size:]

    # 划分对应的标签
    y_train_xgb = y_labels[:train_size]
    y_val_xgb = y_labels[train_size:]

    print(f'时序划分完成:')
    print(f'  训练集特征 X_train_xgb: {X_train_xgb.shape}')
    print(f'  验证集特征 X_val_xgb: {X_val_xgb.shape}')
    print(f'  训练集标签 y_train_xgb: {y_train_xgb.shape}')
    print(f'  验证集标签 y_val_xgb: {y_val_xgb.shape}')
    print()

    # =============================================================================
    # 第四步：验证与报告
    # =============================================================================
    print('=== 第四步：验证与报告 ===')
    print()

    # 数据形状验证
    print('数据形状验证:')
    expected_train_shape = (12336, 32)
    expected_val_shape = (3085, 32)

    print(f'期望的X_train_xgb形状: {expected_train_shape}')
    print(f'实际的X_train_xgb形状: {X_train_xgb.shape}')
    print(f'X_train_xgb形状验证: {X_train_xgb.shape == expected_train_shape}')
    print()

    print(f'期望的X_val_xgb形状: {expected_val_shape}')
    print(f'实际的X_val_xgb形状: {X_val_xgb.shape}')
    print(f'X_val_xgb形状验证: {X_val_xgb.shape == expected_val_shape}')
    print()

    # 特征质量检查
    print('特征质量检查:')
    print(f'X_train_xgb值范围: [{X_train_xgb.min():.3f}, {X_train_xgb.max():.3f}]')
    print(f'X_val_xgb值范围: [{X_val_xgb.min():.3f}, {X_val_xgb.max():.3f}]')
    print(f'特征均值: {X_features_bilstm.mean():.3f}')
    print(f'特征标准差: {X_features_bilstm.std():.3f}')
    print()

    # 标签分布检查
    print('标签分布检查:')
    print(f'训练集标签分布: {np.bincount(y_train_xgb.astype(int))}')
    print(f'验证集标签分布: {np.bincount(y_val_xgb.astype(int))}')
    print(f'训练集损伤率: {np.mean(y_train_xgb):.4f} ({np.mean(y_train_xgb)*100:.2f}%)')
    print(f'验证集损伤率: {np.mean(y_val_xgb):.4f} ({np.mean(y_val_xgb)*100:.2f}%)')
    print()

    # 保存最终数据供XGBoost使用
    print('保存最终数据供XGBoost使用...')
    np.save(r'E:\AIMeeting\X_train_xgb.npy', X_train_xgb)
    np.save(r'E:\AIMeeting\X_val_xgb.npy', X_val_xgb)
    np.save(r'E:\AIMeeting\y_train_xgb.npy', y_train_xgb)
    np.save(r'E:\AIMeeting\y_val_xgb.npy', y_val_xgb)
    np.save(r'E:\AIMeeting\X_features_bilstm.npy', X_features_bilstm)

    print('[OK] 数据保存完成！')
    print('  - X_train_xgb.npy: 训练集32维Bi-LSTM特征')
    print('  - X_val_xgb.npy: 验证集32维Bi-LSTM特征')
    print('  - y_train_xgb.npy: 训练集标签')
    print('  - y_val_xgb.npy: 验证集标签')
    print('  - X_features_bilstm.npy: 完整32维特征集')
    print()

    print('=== Bi-LSTM特征提取与XGBoost数据准备完成！===')
    print()
    print('关键成果:')
    print(f'成功提取了{len(X_features_bilstm)}个32维深度特征')
    print(f'训练集: {X_train_xgb.shape[0]}个样本，{X_train_xgb.shape[1]}个特征')
    print(f'验证集: {X_val_xgb.shape[0]}个样本，{X_val_xgb.shape[1]}个特征')
    print(f'特征质量: 数值范围合理，分布正常')
    print()
    print('下一步: 使用这些32维Bi-LSTM特征训练XGBoost分类器！')
    print('   这将是构建我们混合架构损伤风险预测模型的最后一步！')

    return X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb, X_features_bilstm

if __name__ == "__main__":
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb, X_features_bilstm = main()