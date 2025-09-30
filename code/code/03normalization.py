import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler  # 修正：使用更适合生理数据的RobustScaler

def main():
    print('=== 深度学习数据准备阶段开始 ===')
    print()

    # =============================================================================
    # 第一步：数据归一化 (Data Normalization)
    # =============================================================================
    print('=== 第一步：数据归一化 (Data Normalization) ===')
    print()

    # 加载数据
    print('加载processed_features_real.csv数据...')
    df = pd.read_csv(r'E:\AIMeeting\processed_features_real.csv')

    # 确保Date列正确解析
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    print(f'数据加载完成，形状: {df.shape}')
    print()

    # 识别需要归一化的特征列 - 排除标识符和目标变量
    exclude_cols = ['Player_name', 'Date', 'injury']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print('需要归一化的特征列:')
    for i, col in enumerate(feature_cols, 1):
        print(f'{i:2d}. {col}')
    print(f'总计: {len(feature_cols)}个特征')
    print()

    # 应用RobustScaler归一化 - 专家修正：更适合生理数据
    print('应用RobustScaler归一化（专家修正：更适合生理数据）...')
    print('专家见解：RobustScaler使用中位数和四分位数，对异常值不敏感')
    print('生理数据常含异常值（如生病时的极端指标），RobustScaler更合适')
    scaler = RobustScaler()

    # 提取特征数据进行归一化
    X_features = df[feature_cols].values
    X_normalized = scaler.fit_transform(X_features)

    # 将归一化后的数据更新回DataFrame
    df_normalized = df.copy()
    df_normalized[feature_cols] = X_normalized

    print('归一化完成！')
    print()

    # 验证归一化结果
    print('验证归一化结果:')
    print('归一化后统计摘要:')
    print(df_normalized[feature_cols].describe())
    print()

    # 检查min和max值
    print('特征列最小值和最大值验证:')
    min_values = df_normalized[feature_cols].min()
    max_values = df_normalized[feature_cols].max()

    print('最小值 (应该全部为0.0):')
    print(min_values)
    print()
    print('最大值 (应该全部为1.0):')
    print(max_values)
    print()

    # 验证RobustScaler效果 - 专家修正：不再要求[0,1]区间
    print('验证RobustScaler效果（注意：RobustScaler不保证[0,1]区间）...')
    print(f'特征均值范围: [{df_normalized[feature_cols].mean().min():.3f}, {df_normalized[feature_cols].mean().max():.3f}]')
    print(f'特征标准差范围: [{df_normalized[feature_cols].std().min():.3f}, {df_normalized[feature_cols].std().max():.3f}]')
    print(f'中位数接近0: {np.allclose(df_normalized[feature_cols].median(), 0, atol=0.1)}')
    print()

    # =============================================================================
    # 第二步：构建时序滑动窗口 (Time-Series Windowing)
    # =============================================================================
    print('=== 第二步：构建时序滑动窗口 (Time-Series Windowing) ===')
    print()

    # 定义参数
    WINDOW_SIZE = 32
    print(f'窗口大小 (WINDOW_SIZE): {WINDOW_SIZE}')
    print('使用32天回顾期来预测第33天的损伤风险')
    print()

    # 准备数据 - 分离特征和目标
    X = df_normalized[feature_cols].values
    y = df_normalized['injury'].values

    print(f'特征矩阵X形状: {X.shape}')
    print(f'目标向量y形状: {y.shape}')
    print(f'特征数量: {len(feature_cols)}')
    print()

    # 实现滑动窗口逻辑
    print('构建滑动窗口...')
    X_samples = []
    y_labels = []

    # 按Player_name分组处理
    players = df_normalized['Player_name'].unique()
    print(f'处理运动员数量: {len(players)}')

    sample_count = 0
    skipped_players = 0

    for player in players:
        player_data = df_normalized[df_normalized['Player_name'] == player]
        player_X = player_data[feature_cols].values
        player_y = player_data['injury'].values
        
        n_samples = len(player_data)
        
        # 检查是否有足够的数据
        if n_samples < WINDOW_SIZE + 1:
            skipped_players += 1
            continue
        
        # 为该运动员构建滑动窗口
        for i in range(n_samples - WINDOW_SIZE):
            # 提取32天的特征数据作为输入窗口
            X_window = player_X[i:i + WINDOW_SIZE]
            # 提取第33天的injury值作为预测目标
            y_target = player_y[i + WINDOW_SIZE]
            
            X_samples.append(X_window)
            y_labels.append(y_target)
            sample_count += 1

    print(f'滑动窗口构建完成！')
    print(f'总样本数量: {sample_count}')
    print(f'跳过的运动员数量 (数据不足): {skipped_players}')
    print(f'有效运动员数量: {len(players) - skipped_players}')
    print()

    # =============================================================================
    # 第三步：最终验证与报告
    # =============================================================================
    print('=== 第三步：最终验证与报告 ===')
    print()

    # 转换为NumPy数组
    print('转换为NumPy数组...')
    X_samples = np.array(X_samples)
    y_labels = np.array(y_labels)

    print('转换完成！')
    print()

    # 打印形状
    print('最终数据形状:')
    print(f'X_samples形状: {X_samples.shape}')
    print(f'y_labels形状: {y_labels.shape}')
    print()

    # 验证形状是否符合预期
    expected_shape = (sample_count, WINDOW_SIZE, len(feature_cols))
    print(f'期望的X_samples形状: {expected_shape}')
    print(f'实际的X_samples形状: {X_samples.shape}')
    print(f'形状验证: {X_samples.shape == expected_shape}')
    print()

    # 检查类别平衡
    print('类别分布检查:')
    unique, counts = np.unique(y_labels, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y_labels)) * 100
        print(f'类别 {label}: {count} 样本 ({percentage:.1f}%)')
    print()

    # 数据类型验证
    print('数据类型验证:')
    print(f'X_samples数据类型: {X_samples.dtype}')
    print(f'y_labels数据类型: {y_labels.dtype}')
    print(f'X_samples值范围: [{X_samples.min():.3f}, {X_samples.max():.3f}]')
    print()

    print('=== 深度学习数据准备完成！===')
    print('数据现在已准备好输入到PyTorch Bi-LSTM模型中！')
    print()

    print('数据摘要:')
    print(f'样本总数: {len(X_samples)}')
    print(f'时间窗口长度: {WINDOW_SIZE}')
    print(f'特征数量: {len(feature_cols)}')
    print(f'输入形状: (batch_size, {WINDOW_SIZE}, {len(feature_cols)})')
    print(f'输出形状: (batch_size,)')

    # 保存处理后的数据
    print()
    print('保存处理后的数据...')
    np.save(r'E:\AIMeeting\X_samples.npy', X_samples)
    np.save(r'E:\AIMeeting\y_labels.npy', y_labels)
    
    # 同时保存scaler供后续使用
    import joblib
    joblib.dump(scaler, r'E:\AIMeeting\feature_scaler.pkl')
    
    print('数据已保存为:')
    print('- X_samples.npy (三维时序样本)')
    print('- y_labels.npy (一维标签向量)')
    print('- feature_scaler.pkl (特征归一化器)')
    
    return X_samples, y_labels, feature_cols, scaler

if __name__ == "__main__":
    X_samples, y_labels, feature_cols, scaler = main()