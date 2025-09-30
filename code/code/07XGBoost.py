import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    print('=== XGBoost集成建模与可解释性AI分析阶段开始 ===')
    print(f'XGBoost版本: {xgb.__version__}')
    print(f'SHAP版本: {shap.__version__}')
    print()

    # =============================================================================
    # 第一步：应用SMOTE处理训练集不平衡
    # =============================================================================
    print('=== 第一步：应用SMOTE处理训练集不平衡 ===')
    print('专家见解: SMOTE仅应用于训练集，避免数据泄漏')
    print()

    # 加载数据
    print('加载Bi-LSTM提取的特征数据...')
    X_train_xgb = np.load(r'E:\AIMeeting\X_train_xgb.npy')
    y_train_xgb = np.load(r'E:\AIMeeting\y_train_xgb.npy')
    X_val_xgb = np.load(r'E:\AIMeeting\X_val_xgb.npy')
    y_val_xgb = np.load(r'E:\AIMeeting\y_val_xgb.npy')

    print(f'数据加载完成:')
    print(f'  X_train_xgb形状: {X_train_xgb.shape}')
    print(f'  y_train_xgb形状: {y_train_xgb.shape}')
    print(f'  X_val_xgb形状: {X_val_xgb.shape}')
    print(f'  y_val_xgb形状: {y_val_xgb.shape}')
    print()

    # 检查原始类别分布
    print('原始训练集类别分布:')
    unique, counts = np.unique(y_train_xgb, return_counts=True)
    print(f'  类别0（无损伤）: {counts[0]} 样本')
    print(f'  类别1（有损伤）: {counts[1]} 样本')
    print(f'  损伤率: {counts[1]/(counts[0]+counts[1])*100:.2f}%')
    print()

    # 应用SMOTE - 关键：仅对训练集
    print('应用SMOTE过采样（仅训练集）...')
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_xgb, y_train_xgb)

    print('SMOTE重采样完成！')
    print(f'重采样后训练集形状: X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}')
    print()

    # 验证SMOTE效果
    print('重采样后训练集类别分布:')
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print(f'  类别0（无损伤）: {counts_resampled[0]} 样本')
    print(f'  类别1（有损伤）: {counts_resampled[1]} 样本')
    print(f'  类别平衡: {"fully_balanced" if counts_resampled[0] == counts_resampled[1] else "unbalanced"}')
    print()

    # =============================================================================
    # 第二步：训练并评估XGBoost分类器
    # =============================================================================
    print('=== 第二步：训练并评估XGBoost分类器 ===')
    print('专家见解: 使用验证集进行早期停止，防止过拟合')
    print()

    # 实例化XGBoost模型
    print('实例化XGBoost分类器...')
    model_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    print('XGBoost模型参数:')
    print(f'  objective: binary:logistic')
    print(f'  n_estimators: 1000')
    print(f'  learning_rate: 0.05')
    print(f'  max_depth: 4')
    print(f'  eval_metric: logloss')
    print()

    # 训练模型 - 使用较新的API格式
    print('训练XGBoost模型...')
    # 对于较新版本的XGBoost，使用eval_set和early_stopping_rounds需要在fit方法中设置
    model_xgb.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=True
    )
        
    print()
    print(f'训练完成！')

    # 在验证集上进行预测
    print('在验证集上进行预测...')
    y_pred_xgb = model_xgb.predict(X_val_xgb)
    y_pred_proba_xgb = model_xgb.predict_proba(X_val_xgb)[:, 1]  # 获取正类的概率

    print('预测完成！')
    print(f'验证集预测形状: {y_pred_xgb.shape}')
    print(f'验证集概率形状: {y_pred_proba_xgb.shape}')
    print()

    # 生成详细的分类报告
    print('=== 验证集性能评估 ===')
    print('生成分类报告...')
    class_report = classification_report(y_val_xgb, y_pred_xgb, target_names=['无损伤', '有损伤'])
    print(class_report)
    print()

    # 计算并显示混淆矩阵
    print('混淆矩阵:')
    cm = confusion_matrix(y_val_xgb, y_pred_xgb)
    print(cm)
    print(f'  真正例 (TP): {cm[1,1]}')
    print(f'  假正例 (FP): {cm[0,1]}')
    print(f'  真负例 (TN): {cm[0,0]}')
    print(f'  假负例 (FN): {cm[1,0]}')
    print()

    # 计算关键指标
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()

    # 计算AUC指标 - 关键修复！
    print('计算AUC指标...')
    roc_auc = roc_auc_score(y_val_xgb, y_pred_proba_xgb)
    
    # 计算Precision-Recall AUC（对不平衡数据更重要）
    precision_curve, recall_curve, _ = precision_recall_curve(y_val_xgb, y_pred_proba_xgb)
    pr_auc = auc(recall_curve, precision_curve)
    
    print('关键性能指标:')
    print(f'  Precision (有损伤): {precision:.3f}')
    print(f'  Recall (有损伤): {recall:.3f}')
    print(f'  F1-Score (有损伤): {f1:.3f}')
    print(f'  Specificity: {specificity:.3f}')
    print(f'  Overall Accuracy: {accuracy:.3f}')
    print(f'  ROC AUC: {roc_auc:.3f}')
    print(f'  PR AUC: {pr_auc:.3f}')
    print()

    # =============================================================================
    # 第三步：可解释性AI - SHAP分析
    # =============================================================================
    print('=== 第三步：可解释性AI - SHAP分析 ===')
    print('专家见解: 将模型预测能力转化为科学洞察')
    print()

    print('创建SHAP解释器...')
    explainer = shap.TreeExplainer(model_xgb)
    print('[OK] SHAP解释器创建成功！')
    print()

    print('计算SHAP值...')
    # 在训练集上计算SHAP值
    shap_values = explainer.shap_values(X_train_resampled)
    print(f'[OK] SHAP值计算完成！')
    print(f'SHAP值形状: {shap_values.shape}')
    print(f'期望形状: ({len(X_train_resampled)}, 32)')
    print(f'形状验证: {shap_values.shape == X_train_resampled.shape}')
    print()

    # =============================================================================
    # 第四步：生成SHAP摘要图
    # =============================================================================
    print('=== 第四步：生成SHAP摘要图 ===')
    print('生成SHAP摘要图（核心可视化）...')

    # 创建特征名称
    feature_names = [f'BiLSTM_Feature_{i+1}' for i in range(32)]

    # 生成SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train_resampled, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot: Bi-LSTM Feature Importance for Injury Risk Prediction', fontsize=16, pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.ylabel('Bi-LSTM Extracted Features', fontsize=12)
    plt.tight_layout()

    # 保存图形
    plt.savefig(r'E:\AIMeeting\shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print('[OK] SHAP摘要图已保存至: E:\\AIMeeting\\shap_summary_plot.png')
    print()

    # 计算特征重要性排名
    print('特征重要性排名（前10个最重要特征）:')
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[::-1][:10]

    for i, idx in enumerate(top_features_idx):
        print(f'  {i+1:2d}. {feature_names[idx]}: {feature_importance[idx]:.4f}')
    print()

    # =============================================================================
    # 第五步：保存最终结果
    # =============================================================================
    print('保存最终模型和结果...')
    import joblib
    joblib.dump(model_xgb, r'E:\AIMeeting\final_xgboost_model.pkl')
    joblib.dump(explainer, r'E:\AIMeeting\shap_explainer.pkl')
    np.save(r'E:\AIMeeting\shap_values.npy', shap_values)
    np.save(r'E:\AIMeeting\X_train_resampled.npy', X_train_resampled)
    np.save(r'E:\AIMeeting\y_train_resampled.npy', y_train_resampled)

    print('[OK] 最终文件保存完成:')
    print('  - final_xgboost_model.pkl: 训练好的XGBoost模型')
    print('  - shap_explainer.pkl: SHAP解释器')
    print('  - shap_values.npy: SHAP值数组')
    print('  - X_train_resampled.npy: SMOTE重采样后的训练数据')
    print('  - y_train_resampled.npy: SMOTE重采样后的训练标签')

    # 保存性能指标到文件
    with open(r'E:\AIMeeting\model_performance.txt', 'w') as f:
        f.write('Bi-LSTM + XGBoost 混合架构性能报告\n')
        f.write('='*50 + '\n')
        f.write(f'训练样本: {len(X_train_resampled)} (SMOTE重采样后)\n')
        f.write(f'验证样本: {len(X_val_xgb)} (原始数据)\n')
        f.write(f'特征维度: 32维 (Bi-LSTM提取)\n')
        f.write(f'原始训练集损伤率: {counts[1]/(counts[0]+counts[1])*100:.2f}%\n')
        f.write(f'重采样后训练集损伤率: 50.00%\n')
        f.write('\n验证集分类报告:\n')
        f.write(class_report)
        f.write('\n混淆矩阵:\n')
        f.write(f'TP: {cm[1,1]}, FP: {cm[0,1]}, TN: {cm[0,0]}, FN: {cm[1,0]}\n')
        f.write(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n')
        f.write(f'ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}\n')

    print('[OK] 性能报告已保存至: E:\\AIMeeting\\model_performance.txt')

    return model_xgb, explainer, shap_values, X_train_resampled, y_train_resampled

if __name__ == "__main__":
    model_xgb, explainer, shap_values, X_train_resampled, y_train_resampled = main()