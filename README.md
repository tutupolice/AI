## ⚽ Bi-LSTM + XGBoost 混合架构运动损伤风险预测模型

## 🎯 项目目标 (Project Goal)

本项目旨在解决专业体育领域中**极端类别不平衡**（损伤发生率极低）和**时序依赖性**的挑战，构建一个高敏感度的运动员损伤风险早期预警系统。

核心目标是：利用深度学习从运动员的日常生理和训练负荷数据中提取高维时序特征，并结合集成学习进行准确、可解释的风险分类。

## 💡 核心方法论与创新

本项目采用了行业领先的**混合架构 (Hybrid Architecture)** 和 **时序验证框架 (Time-Series Validation)**：

1.  **Bi-LSTM 特征提取器**：
    * 将原始 $\text{22}$ 维生理指标（如训练负荷、主观疲劳、心率等）输入 **双向长短期记忆网络 ($\text{Bi-LSTM}$)**。
    * 模型从 **$\text{32}$ 天滑动时间窗口** 中学习复杂的疲劳累积模式。
    * 最终输出 **$\text{32}$ 维高层次、抽象化的时序特征**。
2.  **XGBoost 分类器**：
    * 使用 $\text{Bi-LSTM}$ 提取的 $\text{32}$ 维特征训练 **极端梯度提升 ($\text{XGBoost}$)** 分类器进行最终的二分类预测（损伤 $\text{1}$ / 无损伤 $\text{0}$）。
3.  **严格时序验证**：
    * 采用 **$\text{TimeSeriesSplit}$** 进行 **$\text{5}$ 折交叉验证**，确保训练数据在时间上始终早于测试数据，提供贴合实际应用场景的稳定性能评估.
4.  **类别不平衡处理**：
    * 针对原始数据 **$\sim 0.5\%$ 的损伤率**，在 $\text{XGBoost}$ 训练阶段使用了 **$\text{SMOTE}$ (Synthetic Minority Over-sampling Technique)** 对训练集进行过采样，以提高模型对损伤事件的敏感性 ($\text{Recall}$).

## 📊 关键成果与性能指标

基于 $\text{5}$ 折时间序列交叉验证的平均性能指标：

| 指标 (损伤类别: 1) | 平均值 | 标准差 | 临床意义 |
| :--- | :--- | :--- | :--- |
| **F1-Score** | $\text{0.228}$ | $\pm \text{0.091}$ | 综合性能 |
| **Recall (敏感性)** | $\text{0.252}$ | $\pm \text{0.090}$ | 实际损伤事件的检出率 |
| **Precision** | $\text{0.217}$ | $\pm \text{0.096}$ | 阳性预测准确率 |
| **Macro Accuracy** | $\sim \text{0.975}$ | N/A | 整体分类准确率 |

**性能亮点**：模型在维持高整体准确率的同时，成功地实现了 **$\sim \text{25\%}$ 的召回率**，在处理 $\sim \text{0.5\%}$ 损伤率的极端不平衡场景下，证明了混合架构在**早期预警**方面的有效性.

## 🔍 可解释性 AI (SHAP Analysis)

本项目实现了**完全可解释的 $\text{AI}$ 决策**。通过 $\text{SHAP}$ 框架，我们识别了对损伤风险预测贡献最大的抽象特征，并通过相关性分析将其与原始生理指标关联.

### 最重要抽象特征与生理学关联（Top 2）

| 抽象特征 | 平均 $\text{SHAP}$ 值 | 高度相关原始指标 | 生理学洞察 |
| :--- | :--- | :--- | :--- |
| **Feature\_17** | $\text{0.2078}$ | $\text{daily\_load}$ ($\text{0.880}$), $\text{illness}$ ($\text{0.825}$) | **日负荷与生理应激的联合模式** |
| **Feature\_18** | $\text{0.1853}$ | $\text{monotony}$ ($\text{-0.923}$), $\text{soreness}$ ($\text{0.871}$) | **训练单调性与肌肉疲劳的恢复平衡** |

**结论**：$\text{Bi-LSTM}$ 成功地将多维生理信息融合为对损伤风险最敏感的**隐性状态**，为运动科学团队提供了透明的决策支持。

## 📦 项目结构与文件说明

| 文件/目录 | 说明 | 对应代码 |
| :--- | :--- | :--- |
| **数据处理** | | |
| `01loader.py` | 原始数据加载、日期转换与排序 | $\text{01loader.py}$ |
| `02preceded.py` | **特征工程**：特征选择、**分组前向填充 ($\text{ffill}$)** 处理缺失值 | $\text{02preceded.py}$ |
| `03normalization.py` | $\text{MinMaxScaler}$ 归一化、**$\text{32}$ 天时序窗口构建** | $\text{03normalization.py}$ |
| **深度学习 (Bi-LSTM)** | | |
| `04Bi-LSTM.py` | 定义 $\text{Bi-LSTM}$ 架构和**加权损失函数** | $\text{04Bi-LSTM.py}$ |
| `05LSTMsplit_fixed.py` | $\text{Bi-LSTM}$ 模型训练与时序划分 | $\text{05LSTMsplit\_fixed.py}$ |
| `06extra.py` | 使用训练好的 $\text{Bi-LSTM}$ 批量提取 $\text{32}$ 维特征 (`X\_features\_bilstm.npy`) | $\text{06extra.py}$ |
| **集成学习 (XGBoost)** | | |
| `07XGBoost.py` | $\text{SMOTE}$ 处理不平衡、训练 $\text{XGBoost}$、**$\text{SHAP}$ 首次分析** | $\text{07XGBoost.py}$ |
| `08Validation_fixed.py` | **核心验证**：$\text{5}$ 折 $\text{TimeSeriesSplit}$ 交叉验证与最终性能评估 | $\text{08Validation\_fixed.py}$ |
| **报告与可视化** | | |
| `09Result_real.py` | 生成 **$\text{final\_academic\_report\_real.md}$**、相关性热图 | $\text{09Result\_real.py}$ |
| `final\_academic\_report\_real.md` | 最终学术报告 | |
| `correlation\_heatmap\_real.png` | $\text{SHAP}$ 抽象特征与原始指标相关性可视化 | |

---

## 🛠️ 环境依赖 (Dependencies)

本项目基于 $\text{Python 3.x}$ 环境，主要依赖库包括：

* $\text{pandas}$
* $\text{numpy}$
* $\text{scikit-learn}$ ($\text{sklearn}$)
* $\text{pytorch}$ ($\text{torch}$)
* $\text{xgboost}$ ($\text{xgb}$)
* $\text{imblearn}$ ($\text{SMOTE}$)
* $\text{shap}$

## 🚀 运行项目

要完整重现整个项目流程，应按数字顺序依次运行 Python 脚本：

$$\text{01loader.py} \to \text{02preceded.py} \to \text{03normalization.py} \to \text{04Bi-LSTM.py} \to \text{05LSTMsplit\_fixed.py} \to \text{06extra.py} \to \text{07XGBoost.py} \to \text{08Validation\_fixed.py} \to \text{09Result\_real.py}$$
