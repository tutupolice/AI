import pandas as pd
import numpy as np

print('=== 运动科学特征工程阶段开始 ===')
print()

# 加载之前处理的数据
print('加载01loader.py处理后的数据...')
# 修正：使用01loader处理后的CSV文件，保留时序排序结果
df_sorted = pd.read_csv(r'E:\AIMeeting\df_sorted.csv')  # 假设01loader保存了排序结果

# 确保日期列正确解析
df_sorted['Date'] = pd.to_datetime(df_sorted['Date'], errors='coerce')

print(f'数据加载完成，形状: {df_sorted.shape}')
print()

# =============================================================================
# 第一步：特征选择 (Feature Selection)
# =============================================================================
print('=== 第一步：特征选择 (Feature Selection) ===')
print()

# 定义特征子集 - 严格遵循运动科学原则
selected_features = [
    # Identifiers for grouping and target
    'Player_name', 'Date', 'injury', 

    # Subjective Wellness (Internal Load) - 运动员主观感受
    'fatigue', 'mood', 'readiness', 'sleep_duration', 
    'sleep_quality', 'soreness', 'stress', 'illness',

    # Objective Training Load (External Load) - 客观训练负荷
    'atl', 'ctl28', 'ctl42', 'daily_load', 'monotony', 
    'strain', 'weekly_load', 'acwr', 'Average_running_speed', 
    'Top_speed', 'HIR', 'Total_distance', 'avg_heart_rate', 
    'max_heart_rate'
]

print('选择的特征列表:')
for i, feature in enumerate(selected_features, 1):
    print(f'{i:2d}. {feature}')
print()

# 创建新的特征子集DataFrame
df_features = df_sorted[selected_features].copy()

print('特征选择完成！')
print(f'新DataFrame形状: {df_features.shape}')
print()

print('新DataFrame前5行:')
print(df_features.head())
print()

# =============================================================================
# 第二步：缺失值诊断与处理 (Missing Value Handling)
# =============================================================================
print('=== 第二步：缺失值诊断与处理 (Missing Value Handling) ===')
print()

# 诊断现状 - 处理前缺失值统计
print('处理前缺失值统计:')
missing_before = df_features.isnull().sum()
print(missing_before[missing_before > 0])
print(f'总缺失值数量: {missing_before.sum()}')
print()

# 执行分组前向填充 - 最关键的步骤
print('执行分组前向填充...')
print('按Player_name分组，确保每个运动员的缺失值只由其自己的历史数据填充')

# 按Player_name分组并应用前向填充 - 使用现代pandas语法
df_filled = df_features.groupby('Player_name', group_keys=False).apply(
    lambda group: group.ffill()
)

print('分组前向填充完成！')
print()

# 处理初始缺失值 - 专家修正：使用运动科学合理的基线值
print('处理初始缺失值（运动科学基线）...')
print('专家见解：缺失的生理指标不能设为0，应使用合理的基线值')

# 为不同特征设置运动科学合理的基线值
baseline_values = {
    # 主观健康指标 - 正常基线为0（无不适）
    'fatigue': 0, 'mood': 0, 'readiness': 0, 'soreness': 0, 
    'stress': 0, 'illness': 0,
    
    # 睡眠指标 - 使用合理基线
    'sleep_duration': 8.0,  # 成人标准睡眠时长
    'sleep_quality': 4.0,   # 5分制中的良好质量
    
    # 训练负荷指标 - 使用0作为基线（无训练负荷）
    'atl': 0, 'ctl28': 0, 'ctl42': 0, 'daily_load': 0, 
    'monotony': 0, 'strain': 0, 'weekly_load': 0, 'acwr': 1.0,  # ACWR基线为1.0（平衡状态）
    
    # 运动表现指标 - 使用合理基线
    'Average_running_speed': 0, 'Top_speed': 0, 'HIR': 0, 
    'Total_distance': 0, 'avg_heart_rate': 60, 'max_heart_rate': 180
}

# 应用运动科学基线值
df_filled = df_filled.fillna(baseline_values)

print('初始缺失值处理完成（使用运动科学基线）！')
print()

# 最终验证
print('最终验证 - 处理后缺失值统计:')
missing_after = df_filled.isnull().sum()
print(missing_after)
print(f'总缺失值数量: {missing_after.sum()}')
print()

# 验证所有列缺失值是否为0
all_zero_missing = (missing_after == 0).all()
print(f'所有列缺失值都为0: {all_zero_missing}')
print()

print('最终处理完毕的DataFrame前10行:')
print(df_filled.head(10))
print()

print('=== 特征工程阶段完成 ===')
print(f'最终数据集形状: {df_filled.shape}')
print('数据已准备好用于时序窗口构建！')

# 保存处理后的数据供后续使用
df_filled.to_csv(r'E:\AIMeeting\processed_features_real.csv', index=False)
print('特征工程数据已保存至: E:\\AIMeeting\\processed_features_real.csv')