import pandas as pd
import numpy as np

# Step 1: 导入基础库
print('=== Step 1: 导入基础库 ===')
print('pandas版本:', pd.__version__)
print('numpy版本:', np.__version__)
print()

# Step 2: 加载数据集
print('=== Step 2: 加载数据集 ===')
# 使用原始文件路径
df = pd.read_excel(r'E:\AIMeeting\original.xlsx')
print(f'成功加载数据，形状: {df.shape}')
print()

# Step 3: 数据体检
print('=== Step 3: 数据体检 ===')
print('DataFrame.info()结果:')
print(df.info())
print()

# Step 4: 核心时序处理 - 转换日期
print('=== Step 4: 核心时序处理 ===')
print('转换Date列类型...')
print('原始Date列样本:', df['Date'].head())
print('原始Date列类型:', df['Date'].dtype)

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
print('转换后Date列类型:', df['Date'].dtype)
print('转换后Date列样本:', df['Date'].head())
print()

# Step 5: 关键排序
print('=== Step 5: 关键排序 ===')
print('按Player_name和Date进行多级排序...')
df_sorted = df.sort_values(['Player_name', 'Date'], ascending=[True, True])
print('排序完成！')
print('排序后数据形状:', df_sorted.shape)
print()

# Step 6: 验证与报告
print('=== Step 6: 验证与报告 ===')
print('处理后DataFrame前15行:')
print(df_sorted.head(15))
print()

# 额外的验证信息
print('=== 额外验证信息 ===')
print('唯一运动员数量:', df_sorted['Player_name'].nunique())
print('运动员样本:', df_sorted['Player_name'].unique()[:5])
print('日期范围:', df_sorted['Date'].min(), '到', df_sorted['Date'].max())
print('总天数:', (df_sorted['Date'].max() - df_sorted['Date'].min()).days)