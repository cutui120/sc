
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.semi_supervised import LabelSpreading, SelfTrainingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


    # 创建特征名称
    gene_names = [f'Gene_{i+1}' for i in range(n_genes)]
    
    # 创建DataFrame
    df = pd.DataFrame(gene_data, columns=gene_names)
    df['Yield'] = y
    df['Strain_ID'] = [f'Strain_{i+1}' for i in range(n_samples)]
    df['Has_Label'] = df.index < n_labeled
    
    return df, gene_names, true_coeffs

# ==================== 2. 数据预处理 ====================
def preprocess_data(df, gene_names):
    """数据预处理"""
    # 分离有标签和无标签数据
    labeled_mask = df['Has_Label']
    unlabeled_mask = ~labeled_mask
    
    X_labeled = df.loc[labeled_mask, gene_names].values
    y_labeled = df.loc[labeled_mask, 'Yield'].values
    X_unlabeled = df.loc[unlabeled_mask, gene_names].values
    
    # 数据标准化
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    
    # 合并所有数据（用于半监督学习）
    X_all_scaled = np.vstack([X_labeled_scaled, X_unlabeled_scaled])
    
    return {
        'X_labeled': X_labeled_scaled,
        'y_labeled': y_labeled,
        'X_unlabeled': X_unlabeled_scaled,
        'X_all': X_all_scaled,
        'scaler': scaler,
        'gene_names': gene_names
    }

# ==================== 3. 半监督学习模型 ====================
class SemiSupervisedModel:
    def __init__(self, base_model='rf'):
        """初始化半监督模型"""
        self.base_model = base_model
        self.models = {}
        self.scaler = None
        
    def train_with_selftraining(self, X_labeled, y_labeled, X_unlabeled, n_iterations=10):
        """使用自训练（Self-training）的半监督回归"""
        # 选择基础模型
        if self.base_model == 'rf':
            base_regressor = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.base_model == 'gb':
            base_regressor = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
        else:  # ridge
            base_regressor = Ridge(alpha=1.0, random_state=42)
        
        # 使用SelfTrainingRegressor（sklearn的半监督回归）
        self_training_model = SelfTrainingRegressor(
            base_regressor,
            threshold=0.7,  # 置信度阈值
            criterion='threshold',
            verbose=False
        )
        
        # 准备数据
        X_all = np.vstack([X_labeled, X_unlabeled])
        y_all = np.hstack([y_labeled, np.full(len(X_unlabeled), -1)])  # 未标记数据用-1
        
        # 训练
        self_training_model.fit(X_all, y_all)
        
        # 预测所有未标记数据的伪标签
        y_pred_unlabeled = self_training_model.predict(X_unlabeled)
        
        # 合并所有标签用于后续分析
        y_all_predicted = np.hstack([y_labeled, y_pred_unlabeled])
        
        return self_training_model, y_all_predicted
    
    def train_with_graph(self, X_labeled, y_labeled, X_unlabeled, X_all):
        """使用图方法（Label Spreading的变体）"""
        # 为标签传播准备标签向量（未标记数据用-1）
        y_all_for_lp = np.full(len(X_all), -1.0)
        y_all_for_lp[:len(y_labeled)] = y_labeled
        
        # 使用回归问题的标签传播（通过kNN图）
        from sklearn.neighbors import kneighbors_graph
        
        # 构建kNN图
        n_neighbors = min(10, len(X_all) // 3)
        affinity_matrix = kneighbors_graph(
            X_all, n_neighbors=n_neighbors, mode='connectivity', include_self=True
        )
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)  # 对称化
        
        # 标签传播（自定义回归版本）
        # 使用迭代最近邻方法进行伪标签传播
        from sklearn.neighbors import KNeighborsRegressor
        
        knn_reg = KNeighborsRegressor(n_neighbors=5)
        knn_reg.fit(X_labeled, y_labeled)
        
        # 初始伪标签预测
        y_pseudo = knn_reg.predict(X_unlabeled)
        
        # 迭代细化
        for iteration in range(5):
            # 合并数据
            X_combined = np.vstack([X_labeled, X_unlabeled])
            y_combined = np.hstack([y_labeled, y_pseudo])
            
            # 重新训练
            knn_reg = KNeighborsRegressor(n_neighbors=5)
            knn_reg.fit(X_combined, y_combined)
            
            # 更新伪标签
            y_pseudo = knn_reg.predict(X_unlabeled)
        
        y_all_predicted = np.hstack([y_labeled, y_pseudo])
        
        return knn_reg, y_all_predicted
    
    def train_with_pseudo_labeling(self, X_labeled, y_labeled, X_unlabeled, method='ensemble'):
        """使用伪标签增强训练集"""
        # 第一阶段：在有标签数据上训练多个模型
        models = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        # 训练并预测未标记数据
        pseudo_predictions = []
        
        for name, model in models.items():
            model.fit(X_labeled, y_labeled)
            y_pred = model.predict(X_unlabeled)
            pseudo_predictions.append(y_pred)
        
        # 集成伪标签预测（中位数减少异常值影响）
        pseudo_predictions = np.array(pseudo_predictions)
        y_pseudo = np.median(pseudo_predictions, axis=0)
        
        # 第二阶段：使用增强的数据集训练最终模型
        X_augmented = np.vstack([X_labeled, X_unlabeled])
        y_augmented = np.hstack([y_labeled, y_pseudo])
        
        # 使用更强大的模型
        final_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        
        final_model.fit(X_augmented, y_augmented)
        
        y_all_predicted = np.hstack([y_labeled, y_pseudo])
        
        return final_model, y_all_predicted

# ==================== 4. 特征重要性分析 ====================
def analyze_feature_importance(model, gene_names, X_all, y_predicted):
    """分析基因特征的重要性"""
    if hasattr(model, 'feature_importances_'):
        # 随机森林等模型
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 线性模型
        importances = np.abs(model.coef_)
    else:
        # 使用替代方法：基于相关性的重要性
        importances = []
        for i in range(len(gene_names)):
            corr = np.corrcoef(X_all[:, i], y_predicted)[0, 1]
            importances.append(abs(corr))
        importances = np.array(importances)
    
    # 创建特征重要性DataFrame
    feat_importance = pd.DataFrame({
        'Gene': gene_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feat_importance

# ==================== 5. 贝叶斯优化寻找最佳组合 ====================
def optimize_gene_expression(model, scaler, gene_names, data_ranges, n_trials=200):
    """使用贝叶斯优化寻找最佳基因表达组合"""
    
    # 定义优化目标函数
    def objective(trial):
        # 为每个基因生成候选值
        gene_values = []
        for i, gene in enumerate(gene_names):
            # 使用数据范围，但转换到标准化空间
            min_val = data_ranges[i][0]
            max_val = data_ranges[i][1]
            gene_val = trial.suggest_float(f'gene_{i}', min_val, max_val)
            gene_values.append(gene_val)
        
        # 转换为数组并标准化
        X_test = np.array(gene_values).reshape(1, -1)
        X_test_scaled = scaler.transform(X_test)
        
        # 预测产量
        y_pred = model.predict(X_test_scaled)
        
        # 我们希望最大化产量
        return float(y_pred[0])
    
    # 运行贝叶斯优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # 获取最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    # 提取基因表达值
    best_gene_values = []
    for i in range(len(gene_names)):
        best_gene_values.append(best_params[f'gene_{i}'])
    
    return best_gene_values, best_value, study

# ==================== 6. 主执行流程 ====================
def main():
    print("=" * 60)
    print("麦角硫因工程菌株半监督产量预测与优化")
    print("=" * 60)
    
    # 1. 生成/加载数据
    print("\n1. 加载数据...")
    df, gene_names, true_coeffs = generate_simulated_data()
    print(f"   总样本数: {len(df)}, 有标签样本: {df['Has_Label'].sum()}")
    print(f"   基因数量: {len(gene_names)}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    data_dict = preprocess_data(df, gene_names)
    
    # 3. 训练半监督模型
    print("\n3. 训练半监督模型...")
    ss_model = SemiSupervisedModel(base_model='rf')
    
    # 使用伪标签增强方法
    print("   使用伪标签增强方法...")
    model, y_all_predicted = ss_model.train_with_pseudo_labeling(
        data_dict['X_labeled'],
        data_dict['y_labeled'],
        data_dict['X_unlabeled']
    )
    
    # 4. 模型评估
    print("\n4. 模型评估...")
    # 交叉验证（仅使用有标签数据）
    cv_scores = cross_val_score(
        model, data_dict['X_labeled'], data_dict['y_labeled'],
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )
    print(f"   5折交叉验证R²分数: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # 5. 特征重要性分析
    print("\n5. 特征重要性分析...")
    feat_importance = analyze_feature_importance(
        model, gene_names, data_dict['X_all'], y_all_predicted
    )
    
    print("\n   最重要的10个基因:")
    for i, row in feat_importance.head(10).iterrows():
        print(f"   {row['Gene']}: {row['Importance']:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.barh(feat_importance['Gene'][:15], feat_importance['Importance'][:15])
    plt.xlabel('重要性分数')
    plt.title('基因对产量的重要性排名')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('gene_importance.png', dpi=150)
    plt.close()
    
    # 6. 优化最佳基因表达组合
    print("\n6. 优化最佳基因表达组合...")
    
    # 确定每个基因的合理范围（基于现有数据）
    gene_ranges = []
    X_original = np.vstack([
        data_dict['X_labeled'],
        data_dict['X_unlabeled']
    ])
    # 反标准化到原始尺度
    X_original = data_dict['scaler'].inverse_transform(X_original)
    
    for i in range(len(gene_names)):
        gene_min = X_original[:, i].min()
        gene_max = X_original[:, i].max()
        # 稍微扩展范围以探索更广的空间
        range_expansion = 0.2 * (gene_max - gene_min)
        gene_ranges.append((
            gene_min - range_expansion,
            gene_max + range_expansion
        ))
    
    # 使用贝叶斯优化
    best_genes, best_yield, study = optimize_gene_expression(
        model, data_dict['scaler'], gene_names, gene_ranges, n_trials=500
    )
    
    print(f"\n   预测的最高产量: {best_yield:.2f}")
    
    # 7. 保存结果
    print("\n7. 保存结果...")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Gene': gene_names,
        'Optimal_Expression': best_genes,
        'Importance': feat_importance['Importance'].values,
        'Min_Expression': [r[0] for r in gene_ranges],
        'Max_Expression': [r[1] for r in gene_ranges],
        'Recommended_Range_Low': best_genes - 0.1 * np.array([r[1]-r[0] for r in gene_ranges]),
        'Recommended_Range_High': best_genes + 0.1 * np.array([r[1]-r[0] for r in gene_ranges])
    })
    
    # 按重要性排序
    results_df = results_df.sort_values('Importance', ascending=False)
    
    # 保存到CSV
    results_df.to_csv('optimal_gene_expression.csv', index=False)
    
    print("\n   结果已保存到 'optimal_gene_expression.csv'")
    
    # 8. 可视化优化历史
    plt.figure(figsize=(10, 6))
    plt.plot(study.trials_dataframe()['value'], alpha=0.5)
    plt.xlabel('试验次数')
    plt.ylabel('预测产量')
    plt.title('贝叶斯优化历史')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=150)
    plt.close()
    
    # 9. 输出建议
    print("\n" + "=" * 60)
    print("优化建议总结:")
    print("=" * 60)
    
    print("\n▶ 最关键的调节基因（前5名）:")
    for i, row in results_df.head(5).iterrows():
        print(f"   {row['Gene']}: 目标表达量 ≈ {row['Optimal_Expression']:.2f}")
    
    print("\n▶ 实施建议:")
    print("   1. 优先调节前5个重要基因的表达水平")
    print("   2. 在推荐范围内微调以找到实验室条件下的最优值")
    print("   3. 考虑基因间的协同效应，建议:")
    print("      - 同时上调高重要性正相关基因")
    print("      - 同时下调高重要性负相关基因")
    
    print("\n▶ 验证策略:")
    print("   1. 构建3-5个组合进行实验验证")
    print("   2. 以预测最优组合为中心设计响应面实验")
    print("   3. 收集新数据后重新训练模型")
    
    # 10. 生成详细报告
    with open('optimization_report.txt', 'w') as f:
        f.write("麦角硫因工程菌株基因表达优化报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"预测最高产量: {best_yield:.2f}\n")
        f.write(f"交叉验证R²分数: {cv_scores.mean():.3f}\n\n")
        
        f.write("基因表达优化方案:\n")
        f.write("-" * 30 + "\n")
        for i, row in results_df.iterrows():
            f.write(f"{row['Gene']}: {row['Optimal_Expression']:.2f} "
                   f"(范围: {row['Recommended_Range_Low']:.2f}-{row['Recommended_Range_High']:.2f})\n")
    
    print(f"\n   详细报告已保存到 'optimization_report.txt'")
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)
    
    return results_df, best_yield, model

# ==================== 执行主程序 ====================
if __name__ == "__main__":
    # 执行主程序
    results, best_yield, final_model = main()
    
    # 显示部分结果
    print("\n优化后的基因表达量（前10个最重要基因）:")
    print(results.head(10)[['Gene', 'Optimal_Expression', 'Importance']].to_string(index=False))